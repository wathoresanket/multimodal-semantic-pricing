#!/bin/bash
# High-Efficiency Space Deployment Script

# 1. Store current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo "🚀 Preparing lightweight Space deployment..."

# 2. Create a clean 'orphan' branch (no history)
# This removes the 3GB of old Git blobs from the push.
git checkout --orphan hf-space-deploy

# 3. Unstage all files
git reset

# 4. Add only the code and configuration
# Large assets will be downloaded at runtime from wathoresanket/pricing-assets
echo "📦 Adding code-only files..."
git add webapp/ \
        steps/ \
        utils/ \
        config.py \
        requirements.txt \
        predict.py \
        run.sh \
        Dockerfile \
        README.md \
        .gitattributes \
        .gitignore

# 5. Commit the clean state
git commit -m "Deploy to Hugging Face Space (External Assets)"

# 6. Push to Hugging Face (Force update the main branch)
echo "📤 Pushing to Hugging Face Space..."
git push hf hf-space-deploy:main --force

# 7. Clean up: return to original branch and delete temp branch
echo "🔄 Returning to $CURRENT_BRANCH..."
git checkout $CURRENT_BRANCH
git branch -D hf-space-deploy

echo "✅ Done! Your Space is now deploying with a tiny (<1MB) repository."
echo "The first build will take a few minutes as it downloads the assets, but future pushes will be instant."
