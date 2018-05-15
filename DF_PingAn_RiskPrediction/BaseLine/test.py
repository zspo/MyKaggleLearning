import time
def time_convert(timestamp, type):
    #转换成localtime
    time_local = time.localtime(timestamp)
    if type == 'hour':
        #转换成新的时间格式(2016-05-05 20:28:54)
        # dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
        dt = time.strftime("%H", time_local)
    else:
        dt = time.strftime("%w", time_local)
    return dt

timestamp = 1477272480
dt = time_convert(timestamp, 'week')
print(type(dt))