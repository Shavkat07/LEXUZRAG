#!/usr/bin/env bash
# scripts/setup_nginx.sh
# Скрипт для настройки Nginx в качестве Reverse Proxy для Lex.uz Legal RAG

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NGINX_CONF="${PROJECT_DIR}/nginx/lexuz.conf"

echo "=== [1/4] Копирование конфигурации Nginx ==="
cp "${NGINX_CONF}" /etc/nginx/sites-available/lexuz

echo "=== [2/4] Активация сайта в Nginx ==="
mkdir -p /etc/nginx/sites-enabled
ln -sf /etc/nginx/sites-available/lexuz /etc/nginx/sites-enabled/lexuz
if [ -f /etc/nginx/sites-enabled/default ]; then
    echo "Отключение дефолтного сайта (default)..."
    rm -f /etc/nginx/sites-enabled/default
fi

echo "=== [3/4] Настройка брандмауэра (UFW) ==="
if command -v ufw >/dev/null 2>&1; then
    ufw allow 80/tcp || true
    ufw allow 8000/tcp || true
fi

echo "=== [4/4] Проверка конфигурации и перезапуск Nginx ==="
nginx -t
systemctl restart nginx
systemctl enable nginx

echo "✅ Nginx успешно настроен и запущен!"
echo "Теперь локально сайт доступен по адресу: http://localhost"
