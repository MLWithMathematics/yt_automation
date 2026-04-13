import sys, os
dest = r'D:\Ai Agents\YT Automation\agents\video_agent.py'
content = sys.stdin.read()
with open(dest, 'w', encoding='utf-8') as f:
    f.write(content)
print('Done, bytes:', len(content))