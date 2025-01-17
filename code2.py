from collections import defaultdict

def calculate_rostering_days(n, m, friendships, k):
    
    friends = defaultdict(list)
    for a, b in friendships:
        friends[a].append(b)
        friends[b].append(a)

    
    current_status = [1] * n
    total_rostering = n
    days = 1 

    while total_rostering < k:
        next_status = [0] * n  
        for emp in range(n):
            count_friends_wfo = sum(current_status[friend] for friend in friends[emp])

            
            if current_status[emp] == 1: 
                next_status[emp] = 1 if count_friends_wfo == 3 else 0
            else:  
                next_status[emp] = 1 if count_friends_wfo < 3 else 0

       
        current_status = next_status
        total_rostering += sum(current_status)
        days += 1

    return days

n, m = map(int, input().split())
friendships = [tuple(map(int, input().split())) for _ in range(m)]
k = int(input())

print(calculate_rostering_days(n, m, friendships, k))
