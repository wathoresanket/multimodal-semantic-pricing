"""
Explainability Module — Text SHAP + Image GradCAM.

Text: Word-level occlusion-based SHAP values.
Image: GradCAM on SigLIP-2 ViT's last encoder layer.
"""
import sys, os, io, base64, re
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config


def compute_text_shap(predictor, text: str, image: Image.Image, img_emb: np.ndarray, baseline_price: float) -> list:
    """
    Compute word-level importance by masking each word and measuring price change.
    
    Returns: list of {word, importance, direction} sorted by |importance|
    """
    # Clean and tokenize
    cleaned = re.sub(r'(Item Name:|Bullet Point \d+:|Product Description:|Value:|Unit:)', ' ', text, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    words = cleaned.split()

    if len(words) == 0:
        return []

    # Limit to first 30 words for speed
    words = words[:30]
    importances = []

    for i, word in enumerate(words):
        masked_words = words[:i] + words[i+1:]
        masked_text = ' '.join(masked_words)

        # Predict with word removed
        masked_price = predictor.predict_with_custom_text(
            f"Item Name: {masked_text}", image, img_emb
        )

        # Importance = how much price changes when word is removed
        # Positive = word increases price, Negative = word decreases price
        importance = baseline_price - masked_price

        importances.append({
            "word": word,
            "importance": round(importance, 4),
            "direction": "increase" if importance > 0 else "decrease",
        })

    # Sort by absolute importance
    importances.sort(key=lambda x: abs(x["importance"]), reverse=True)
    return importances


def compute_gradcam(predictor, image: Image.Image) -> str:
    """
    Compute GradCAM heatmap using SigLIP-2's last encoder layer.
    
    Returns: base64-encoded PNG of the heatmap overlay.
    """
    model = predictor.img_model
    processor = predictor.img_processor

    # Prepare input
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}

    # Enable gradients for this pass
    model.eval()
    
    # Hook into the last encoder layer
    activations = {}
    gradients = {}

    def save_activation(name):
        def hook(module, input, output):
            # output is a tuple for encoder layers
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            else:
                activations[name] = output.detach()
        return hook

    def save_gradient(name):
        def hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                gradients[name] = grad_output[0].detach()
            else:
                gradients[name] = grad_output.detach()
        return hook

    # Register hooks on the last encoder layer
    last_layer = model.vision_model.encoder.layers[-1]
    fwd_hook = last_layer.register_forward_hook(save_activation('last'))
    bwd_hook = last_layer.register_full_backward_hook(save_gradient('last'))

    try:
        # Forward pass with gradients
        for param in model.parameters():
            param.requires_grad_(False)
        
        pixel_values = inputs['pixel_values'].clone().requires_grad_(True)
        modified_inputs = {k: v for k, v in inputs.items()}
        modified_inputs['pixel_values'] = pixel_values
        
        out = model(**modified_inputs)
        
        # Use the pooler output norm as the target
        target = out.pooler_output.norm()
        
        # Enable gradients on the activation for backward
        if 'last' in activations:
            activations['last'].requires_grad_(True)
        
        target.backward(retain_graph=True)

        if 'last' not in activations or 'last' not in gradients:
            # Fallback: return plain image if hooks failed
            return _image_to_base64(image)

        act = activations['last']   # [1, num_patches, hidden_dim]
        grad = gradients['last']    # [1, num_patches, hidden_dim]

        # GradCAM: weight activations by gradients
        weights = grad.mean(dim=-1, keepdim=True)  # [1, num_patches, 1]
        cam = (weights * act).sum(dim=-1).squeeze()  # [num_patches]

        # Remove CLS token if present
        num_patches = cam.shape[0]
        img_size = inputs['pixel_values'].shape[-1]  # 384
        patch_size = 14
        grid_size = img_size // patch_size  # 27

        expected_patches = grid_size * grid_size
        if num_patches == expected_patches + 1:
            cam = cam[1:]  # Remove CLS
        elif num_patches > expected_patches:
            cam = cam[:expected_patches]

        # Reshape to grid
        actual_grid = int(np.sqrt(cam.shape[0]))
        if actual_grid * actual_grid != cam.shape[0]:
            actual_grid = int(np.ceil(np.sqrt(cam.shape[0])))
            cam = cam[:actual_grid * actual_grid]
        
        cam = cam.reshape(actual_grid, actual_grid).detach().cpu().numpy()

        # Normalize to [0, 1]
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

    finally:
        fwd_hook.remove()
        bwd_hook.remove()
        # Re-disable gradients
        for param in model.parameters():
            param.requires_grad_(False)

    # Create overlay
    return _create_overlay(image, cam)


def _create_overlay(image: Image.Image, cam: np.ndarray) -> str:
    """Overlay GradCAM heatmap on image, return base64 PNG."""
    img_resized = image.resize((384, 384))
    img_array = np.array(img_resized).astype(np.float32) / 255.0

    # Resize CAM to image size
    from PIL import Image as PILImage
    cam_pil = PILImage.fromarray((cam * 255).astype(np.uint8))
    cam_resized = cam_pil.resize((384, 384), PILImage.BILINEAR)
    cam_array = np.array(cam_resized).astype(np.float32) / 255.0

    # Apply colormap
    heatmap = cm.jet(cam_array)[:, :, :3]

    # Overlay
    overlay = 0.5 * img_array + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(overlay)
    ax.axis('off')
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode('utf-8')


def _image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string."""
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
