# Parser for Ph2ACF Log
# To check the amount of time taken by each Ph2ACF task
# Author : Gourab Saha
# Assistant : ChatGPT :)


#!/bin/bash

# Prompt user for start and stop times
read -p "Calib Dir: " CALIBDIR
read -p "Enter START FROM (dd.mm.yyyy hh:mm:ss): " START_FROM
read -p "Enter STOP AT   (dd.mm.yyyy hh:mm:ss): " STOP_AT

LOGFILE="$CALIBDIR/Ph2_ACF/logs/Ph2_ACF.log"

echo "Initialized : $START_FROM"

gawk -v start_from="$START_FROM" -v stop_at="$STOP_AT" '
BEGIN {
    start_filter = to_epoch(start_from)
    stop_filter  = to_epoch(stop_at)

    format = "%-30s | %-19s | %-19s | %13s\n"

    printf format, "TASK", "START", "STOP", "DURATION (hh:mm:ss)"
    print "----------------------------------------------------------------------------------------------"

    total_duration = 0
}

{
    clean_time = $2
    sub(/:$/, "", clean_time)   # remove trailing colon

    timestamp = $1 " " clean_time
    line_epoch = to_epoch(timestamp)

    if (line_epoch < start_filter)
        next

    if (line_epoch > stop_filter)
        exit
}

/Starting */ {
    task=$0
    sub(/^.*Starting /,"",task)
    sub(/ (measurement|Equalization)(\..*)?$/, "", task)
    
    starts[task] = timestamp
    start_epoch[task] = line_epoch

    if (!first_start) first_start = line_epoch  # track first START
}

/Stopping */ {
    task=$0
    sub(/^.*Stopping /,"",task)
    sub(/ (measurement|Equalization)(\..*)?$/, "", task)

    if (task in start_epoch) {
        duration = line_epoch - start_epoch[task]
    	d_hours = int(duration / 3600)
    	d_minutes = int((duration % 3600) / 60)
    	d_seconds = duration % 60

	duration_str = sprintf("%02d:%02d:%02d",
                       d_hours, d_minutes, d_seconds)

	total_duration += duration
        last_stop = line_epoch  # track last STOP

        printf format,
               task,
               starts[task],
               timestamp,
               duration_str
    }
}

END {
    print "----------------------------------------------------------------------------------------------"

    # Convert total_duration (sum of DURATION column) to HH:MM:SS
    td = total_duration
    td_hours = int(td / 3600)
    td_minutes = int((td % 3600) / 60)
    td_seconds = td % 60
    printf "Sum of DURATION column : %02d:%02d:%02d\n", td_hours, td_minutes, td_seconds

    # Total time from first start to last stop
    total_time = last_stop - first_start
    hours = int(total_time / 3600)
    minutes = int((total_time % 3600) / 60)
    seconds = total_time % 60
    printf "Total duration (first START to last STOP) : %02d:%02d:%02d\n", hours, minutes, seconds
}

function to_epoch(datetime,   d,t,dp) {
    split(datetime, dt, " ")
    d = dt[1]
    t = dt[2]

    split(d, dp, ".")
    return mktime(dp[3]" "dp[2]" "dp[1]" " substr(t,1,2)" "substr(t,4,2)" "substr(t,7,2))
}
' "$LOGFILE"

echo "Stopped : $STOP_AT"
