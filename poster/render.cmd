@echo off
cd /d %~dp0
set CH="C:\Program Files\Google\Chrome\Application\chrome.exe"
set U=file:///%CD:\=/%/poster.html
%CH% --headless=new --disable-gpu --hide-scrollbars --window-size=2290,3224 --screenshot="%CD%\shot.png" %U% >nul 2>&1
%CH% --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="%CD%\poster.pdf" %U% >nul 2>&1
