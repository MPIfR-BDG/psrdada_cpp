for ii in $(ipcs -m | grep "0x" | awk '{print $1}'); do ipcrm -M $ii; done
ipcs -s | grep root | awk ' { print $2 } ' | xargs ipcrm sem