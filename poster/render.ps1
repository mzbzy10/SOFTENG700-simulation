$ErrorActionPreference="Continue"
cd $PSScriptRoot
$ch="C:\Program Files\Google\Chrome\Application\chrome.exe"
$u="file:///$($PWD.Path.Replace('\','/'))/poster.html"
& $ch --headless=new --disable-gpu --hide-scrollbars --window-size=2290,3224 --screenshot="$PWD\shot.png" $u 2>$null
& $ch --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$PWD\poster.pdf" $u 2>$null
