#Async connection pool
ASYNC_MIN_POOL = 2
ASYNC_MAX_POOL = 5
POOL_TIMEOUT_SECONDS = 30 #cả sync lẫn async pool = default ngầm của psycopg

#Sync connection pool
SYNC_MIN_POOL = 2
SYNC_MAX_POOL = 10