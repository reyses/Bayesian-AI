#!/bin/bash
HOST_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
curl http://$HOST_IP:11435/api/tags
