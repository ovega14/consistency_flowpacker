#!/bin/bash

python3 -m ../src
exit_code=$?

if [[ "$exit_code" != "0" ]]; then
    echo "Some tests failed!"
fi

exit $exit_code

