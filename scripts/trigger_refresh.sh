#!/bin/zsh
# Poke the ai-money-flow refresh workflow during US market hours (ET, Mon-Fri).
# Supplements GitHub's own (throttled) cron; harmless if both fire.
export TZ=America/New_York
day=$(date +%u); hm=$(date +%H%M)
[[ $day -le 5 ]] || exit 0
[[ $hm -ge 0925 && $hm -le 1605 ]] || exit 0
/opt/homebrew/bin/gh workflow run refresh.yml --repo LehmanTrader/ai-money-flow 2>&1 | logger -t flowrefresh
