#ifndef SRC_SHARED_SHAREDLOCK_H_
#define SRC_SHARED_SHAREDLOCK_H_

#include <stdint.h>

int64_t sharedlock_size();

void sharedlock_create(char* data);

void sharedlock_acquire(char* data);

void sharedlock_release(char* data);

void sharedlock_destroy(char* data);

#endif /* SRC_SHARED_SHAREDLOCK_H_ */
