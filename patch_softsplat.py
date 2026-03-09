import re

with open("src/models/generalizable_INR/modules/softsplat.py", "r") as f:
    code = f.read()

# Let's just rewrite the whole file with the clamp fixes applied directly, this avoids annoying regex.
