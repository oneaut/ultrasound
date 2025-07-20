# 1. Initialize Git (if you haven’t already)
git init

# 2. Stage all files
git add .

# 3. Commit them
git commit -m "Initial commit: add beamforming scripts and README"

# 4. Link to your GitHub repo
#    Replace <USERNAME> and <REPO> with your account and repo names.
#    You can copy this URL from the “…or push an existing repository” section on GitHub.
git remote add origin https://github.com/<USERNAME>/<REPO>.git

# 5. Push up your default branch (often 'main')
git branch -M main
git push -u origin main
