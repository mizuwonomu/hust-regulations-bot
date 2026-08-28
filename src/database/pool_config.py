#Async connection pool
ASYNC_MIN_POOL = 2
ASYNC_MAX_POOL = 5
POOL_TIMEOUT_SECONDS = 30 #cả sync lẫn async pool = default ngầm của psycopg

#Sync connection pool
SYNC_MIN_POOL = 2
SYNC_MAX_POOL = 10

# Mỗi lần thử getconn <= 0.05s trong thread (thread nhả token),
# giữa hai lần thử nhường loop 0.01s; tổng thời gian chờ vẫn bám timeout pool
CONN_POLL_INTERVAL = 0.05
CONN_POLL_RETRY_DELAY = 0.01