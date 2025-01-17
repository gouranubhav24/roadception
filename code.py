from collections import Counter
def ritika_string_task(n, substrings, main_string, k):
    def remove_unwanted(sub, target, remaining_deletions):
        sub_count = Counter(sub)
        target_count = Counter(target)
        deletions = 0

        for char in sub_count:
            if sub_count[char] > target_count[char]:
                deletions += sub_count[char] - target_count[char]
                sub_count[char] = target_count[char]
            if deletions > remaining_deletions:
                return None, remaining_deletions

        result = ""
        for char in sub:
            if sub_count[char] > 0:
                result += char
                sub_count[char] -= 1

        return result, remaining_deletions - deletions

    max_len = len(main_string)
    formed_string = ""
    used_deletions = 0

    i = 0
    while i < len(main_string):
        found = False
        for sub in substrings:
            sub_result, deletions_left = remove_unwanted(sub, main_string[i:], k - used_deletions)
            if sub_result is not None and main_string[i:].startswith(sub_result):
                formed_string += sub_result
                i += len(sub_result)
                used_deletions += (k - deletions_left)
                found = True
                break

        if not found:
            break

    if formed_string == main_string and used_deletions <= k:
        return "Possible"
    elif len(formed_string) == 0:
        return "Nothing"
    elif len(formed_string) < len(main_string):
        return formed_string
    return "Impossible"

# Input reading
n = int(input())
substrings = [input().strip() for _ in range(n)]
main_string = input().strip()
k = int(input())

# Output the result
print(ritika_string_task(n, substrings, main_string, k))
