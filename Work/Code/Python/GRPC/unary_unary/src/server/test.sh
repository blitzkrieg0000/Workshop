#!/bin/bash
python main_server.py &
echo "Server Başlatılıyor..."
sleep 3
echo "Server Başlatıldı..."
python main_client.py
