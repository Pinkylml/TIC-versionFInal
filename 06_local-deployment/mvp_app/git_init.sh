#!/bin/bash
# Initialize Git
git init

# Add Remote (User specified git@github.com-personal:Pinkylml/MVP-TIC.git)
git remote add origin git@github.com-personal:Pinkylml/MVP-TIC.git

# Create .gitignore
echo "__pycache__/" > .gitignore
echo "venv/" >> .gitignore
echo ".env" >> .gitignore
echo ".DS_Store" >> .gitignore
echo "mvp_outputs/" >> .gitignore

# Add files
git add .

# First commit
git commit -m "Initial MVP commit: XGB-AFT Survival App"

# Push (Force or normal? remote might not exist or be empty)
# User asked for commands, running this might fail if no SSH key or repo exists. 
# I will output the command instruction in the notify message mostly, but this script is the artifact.
echo "Git setup complete. Run 'git push -u origin master' (or main) to push."
