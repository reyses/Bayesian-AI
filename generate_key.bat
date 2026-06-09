@echo off
mkdir %USERPROFILE%\.ssh 2>nul
ssh-keygen -t rsa -b 4096 -f %USERPROFILE%\.ssh\id_rsa -N "" -q
type %USERPROFILE%\.ssh\id_rsa.pub
