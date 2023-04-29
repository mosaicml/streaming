#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>

#include "sharedlock.h"

int64_t sharedlock_size() {
    return sizeof(pthread_mutex_t);
}

void sharedlock_create(char* data) {
    pthread_mutex_t* lock = (pthread_mutex_t*)data;
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setpshared(&attr, 1);
    pthread_mutex_init(lock, &attr);
}

void sharedlock_acquire(char* data) {
    pthread_mutex_t* lock = (pthread_mutex_t*)data;
    pthread_mutex_lock(lock);
}

void sharedlock_release(char* data) {
    pthread_mutex_t* lock = (pthread_mutex_t*)data;
    pthread_mutex_unlock(lock);
}

void sharedlock_destroy(char* data) {
    pthread_mutex_t* lock = (pthread_mutex_t*)data;
    pthread_mutex_destroy(lock);
}
