export STORAGE_LOCATION=$(pwd)/anythingllm && \
sudo mkdir -p $STORAGE_LOCATION && \
sudo touch "$STORAGE_LOCATION/.env" && \
sudo docker run -d -p 3001:3001 \
--cap-add SYS_ADMIN \
-v ${STORAGE_LOCATION}:/app/server/storage \
-v ${STORAGE_LOCATION}/.env:/app/server/.env \
-e STORAGE_DIR="/app/server/storage" \
mintplexlabs/anythingllm