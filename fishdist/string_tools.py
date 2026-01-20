import os


# Simple implementation of the Levenshtein distance algorithm
def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


# Function to extract the base name (removes path and suffixes like (1))
def _extract_base_name(filename):
    base_name = os.path.basename(filename)
    base_name = os.path.splitext(base_name)[0]
    # base_name = re.sub(r'\(\d+\)$', '', base_name)
    return base_name

def longest_common_substring(strings, use_base_name=False):
    # If no strings, return an empty string
    if not strings:
        return ""

    if use_base_name:
        strings = [_extract_base_name(string) for string in strings]

    # Get the shortest string as a starting point for comparison
    shortest_str = min(strings, key=len)

    # Iterate over the substring lengths starting from the full length of the shortest string
    for i in range(len(shortest_str), 0, -1):
        for j in range(len(shortest_str) - i + 1):
            # Get a candidate substring from the shortest string
            substring = shortest_str[j:j + i]

            # Check if this substring is present in all other strings
            if all(substring in string for string in strings):
                return substring

    return ""