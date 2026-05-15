echo # Conditional DDPM > README.md
echo. >> README.md
echo ## Results >> README.md
echo - test.json accuracy: 0.7361 ^(90%^) >> README.md
echo - new_test.json accuracy: 0.8810 ^(100%^) >> README.md
echo. >> README.md
echo ## Architecture >> README.md
echo - Conditional UNet with linear noise schedule >> README.md
echo - EMA weights for inference >> README.md
echo - Gradient accumulation ^(effective batch size 32^) >> README.md
echo - Self-attention at bottleneck only >> README.md
git add README.md
git commit -m "Add README"
git push
