#!/bin/bash

# 检查是否提供了参数
if [ -z "$1" ]; then
    echo "Usage: $0 {start|stop}"
    exit 1
fi

# 根据参数执行相应的操作
case "$1" in
    start)
        # 切换到指定目录并启动 Docker 容器
        cd ./ExecEval
        docker run -it -p 5000:5000 -e NUM_WORKERS=37 exec-eval:1.0
        ;;
    stop)
        # 获取正在运行的 exec-eval:1.0 容器的容器ID
        container_id=$(docker ps -q --filter ancestor=exec-eval:1.0)

        # 如果容器正在运行，则停止它
        if [ -n "$container_id" ]; then
            docker stop $container_id
            echo "Stopped container with ID: $container_id"
        else
            echo "No running container found for image exec-eval:1.0"
        fi
        ;;
    *)
        echo "Invalid option: $1"
        echo "Usage: $0 {start|stop}"
        exit 1
        ;;
esac
