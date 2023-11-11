#!/bin/bash

total_duration=0

# 遍历文件夹下的所有m4a文件
for file in xmly/*.mp3; do
    # 使用FFmpeg获取音频时长信息（以秒为单位）
    duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file")
    
    # 将音频时长累加到总时长中
    total_duration=$(echo "$total_duration + $duration" | bc -l)
done

# 转换为整数部分和小数部分
total_duration_int=$(echo "$total_duration" | cut -d '.' -f 1)
total_duration_frac=$(echo "$total_duration" | cut -d '.' -f 2)

# 输出总音频时长（以小时:分钟:秒的格式）
printf "总音频时长: %02d:%02d:%02d.%s\n" $((total_duration_int/3600)) $((total_duration_int%3600/60)) $((total_duration_int%60)) "$total_duration_frac"