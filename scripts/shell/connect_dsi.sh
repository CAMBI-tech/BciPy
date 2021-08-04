#!/bin/bash
###### Connect DSI LSL. #######
# Make sure DSI cap is on and paired via bluetooth or connection to a port.
# You can locate what port is paired to the device in the Device Manager in Windows.
# Change directory to the location of the DSI App

sname_prompt="Select the DSI headset type:"
options=("DSI" "DSI_VR300")
echo "DSI headset types"
PS3="$sname_prompt "
select opt in "${options[@]}"; do
case $REPLY in
1 ) sname="$opt"; break;;
2 ) sname="$opt"; break;;
*) echo "Invalid option. Try another one (Ctrl+C to exit)."; continue;;
esac
done

read -p "Enter a COM port [COM3]: " com
com=${com:-COM3}

read -rsp $"Press Enter to Connect to $sname"
printf "\nStarting connection to $sname on $com.. Press Ctrl+C to end session."
eval "./dsi2lsl.exe --lsl-stream-name=$sname --port=$com"

read -rsp $"Error. Please address messages above. Press enter to exit."
printf "disconnected"
