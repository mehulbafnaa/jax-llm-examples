#!/bin/bash
#
# This script should be run on host-1 to authorize SSH connections from host-0.
# It adds host-0's public key to host-1's authorized_keys file.

# Ensure the .ssh directory exists with the correct permissions
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# The public key from host-0
PUBLIC_KEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILClVMhM/zXpvJ3k8FUxHrLaZT5CyLtfsOfSrltAnWmK mehulbafna@t1v-n-0b35dafc-w-0"

# Add the key to authorized_keys and ensure the file has the correct permissions
echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

echo "Host-0's public key has been added to host-1's authorized_keys."
echo "You should now be able to SSH from host-0 to host-1 without a password."
