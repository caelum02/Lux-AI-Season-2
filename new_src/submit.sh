#!/bin/bash
rm -r lux/__pycache__ 
tar -cvzf submission.tar.gz action_enum.py agent.py main.py lux/*
