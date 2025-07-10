@echo off
echo Pushing Image Sharpening Project to GitHub...

REM Initialize git if not already done
git init

REM Add remote repository
git remote add origin https://github.com/sanatanisher01/image-sharpening.git

REM Add all files
git add .

REM Commit changes
git commit -m "Complete Image Sharpening System with Knowledge Distillation for Video Conferencing"

REM Push to GitHub
git push -u origin main

echo Done! Project pushed to GitHub.
pause