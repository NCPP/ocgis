#!/usr/bin/env bash

# Launch the test server.
fab test_node_launch

# Run tests on the launched server.
fab test_node_run_tests:branch=next,failed=false