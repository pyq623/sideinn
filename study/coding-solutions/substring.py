def lengthOfLongestSubstring(s: str) -> int:
    i = -1
    res = 0
    hash_dic = {}
    for j in range(len(s)):
        if s[j] in hash_dic: # 重复出现了
            i = max(i, hash_dic[s[j]])
        res = max(res, (j - i)) # 每次都比较，因为可能出现从头到尾都没有重复的情况
        hash_dic[s[j]] = j
    return res
