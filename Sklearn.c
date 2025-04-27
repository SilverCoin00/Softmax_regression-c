#pragma once
#include "Pandas_&_Numpy.c"
#include <math.h>
#include <time.h>

static struct Space {
    int key;
	char* keys;
	struct Space* next;
};
typedef struct Space Space;
static struct Set {
    Space** arr;
	unsigned max_size;
	int size;
};
typedef struct Set Set;

typedef struct Standard_scaler {
    int features;
    float* mean;
    float* deviation;
} Standard_scaler;
typedef struct Min_max_scaler {
    int features;
    float* min;
    float* max;
} Min_max_scaler;
typedef struct One_hot_encoder {
    Set* sample_types;
} One_hot_encoder;
typedef struct Label_encoder {
    Set* sample_types;
} Label_encoder;
typedef struct Simple_Imputer {
    float* digit_data;
    char** str_data;
} Simple_Imputer;

static int is_prime(int n) {
	if(n < 2) return 0;
    for(int i = 2; i*i <= n; i++){
        if(n % i == 0) {
            return 0;
        }
    }
    return 1;
}
static void trans_prime(int* n) {
	if (*n % 2 == 0) (*n)++;
	while (!is_prime(*n)) *n += 2;
}
static int hash_func(const char* keys, int max_size) {
    if (!keys) return 0;
	int hash = 0;
	while (*keys) hash = (hash* 31) + (*keys++);
	return hash % max_size;
}
static Set* new_set(int max_size) {
	Set* st = (Set*)malloc(sizeof(Set));
	trans_prime(&max_size);
	st->max_size = max_size;
	st->size = 0;
	st->arr = (Space**)malloc(st->max_size* sizeof(Space*));
	for (int i = 0; i < st->max_size; i++) {
		st->arr[i] = NULL;
	}
	return st;
}
static void set_add(Set* st, const char* keys, int key) {
	int id = hash_func(keys, st->max_size);
	Space* head = st->arr[id];
	Space* mid = head;
	while (mid != NULL) {
		if (!strcmp(mid->keys, keys)) return ;
		mid = mid->next;
	}
	Space* news = (Space*)malloc(sizeof(Space));
	news->keys = strdup(keys);
    news->key = key;
	news->next = head;
	st->arr[id] = news;
	st->size++;
}
static int set_find(Set* st, const char* keys) {
    int id = hash_func(keys, st->max_size);
    Space* mid = st->arr[id];
    while (mid != NULL) {
        if (!strcmp(mid->keys, keys)) return 1;
        mid = mid->next;
	}
    return 0;
}
static int set_call(Set* st, const char* keys) {
    int id = hash_func(keys, st->max_size);
    Space* mid = st->arr[id];
    while (mid != NULL) {
        if (!strcmp(mid->keys, keys)) return mid->key;
        mid = mid->next;
	}
    return -1;
}
static int* set_key_access(Set* st, const char* keys) {
    int id = hash_func(keys, st->max_size);
    Space* mid = st->arr[id];
    while (mid != NULL) {
        if (!strcmp(mid->keys, keys)) return &(mid->key);
        mid = mid->next;
	}
    return NULL;
}
static void free_set(Set* st) {
	if (!st) return ;
    Space* mid, * next;
	for (int i = 0; i < st->max_size; i++) {
        mid = st->arr[i];
        while (mid != NULL) {
            next = mid->next;
            if (mid->keys) free(mid->keys);
            free(mid);
            mid = next;
        }
    }
    free(st->arr);
    free(st);
}

void* new_scaler(char* scaler_type) {
    if (!strcmp(scaler_type, "Standard_scaler")) {
        Standard_scaler* scaler = (Standard_scaler*)malloc(sizeof(Standard_scaler));
        return (void*) scaler;
    } else if (!strcmp(scaler_type, "Min_max_scaler")) {
        Min_max_scaler* scaler = (Min_max_scaler*)malloc(sizeof(Min_max_scaler));
        return (void*) scaler;
    }
    return NULL;
}
void scaler_fit(float** x, float* y, int samples, int features, void* scaler, char* scaler_type) {
    if (!strcmp(scaler_type, "Standard_scaler")) {
        Standard_scaler* scl = (Standard_scaler*)scaler;
        scl->features = features + 1;
        scl->mean = (float*)malloc(scl->features* sizeof(float));
        scl->deviation = (float*)malloc(scl->features* sizeof(float));
        float sum;
        int i, j;
        for (i = 0; i < scl->features - 1; i++) {
            sum = 0;
            for (j = 0; j < samples; j++) sum += x[j][i];
            scl->mean[i] = sum / samples;
            sum = 0;
            for (j = 0; j < samples; j++) sum += (scl->mean[i] - x[j][i])* (scl->mean[i] - x[j][i]);
            scl->deviation[i] = sqrt(sum / (samples - 1));
        }
        if (y) {
            for (i = 0, sum = 0; i < samples; i++) sum += y[i];
            scl->mean[scl->features - 1] = sum / samples;
            for (i = 0, sum = 0; i < samples; i++) 
                sum += (scl->mean[scl->features - 1] - y[i])* (scl->mean[scl->features - 1] - y[i]);
        }
        scl->deviation[scl->features - 1] = sqrt(sum / (samples - 1));
    } else if (!strcmp(scaler_type, "Min_max_scaler")) {
        Min_max_scaler* scl = (Min_max_scaler*)scaler;
        scl->features = features + 1;
        scl->min = (float*)malloc(scl->features* sizeof(float));
        scl->max = (float*)malloc(scl->features* sizeof(float));
        int i, j;
        for (i = 0; i < scl->features - 1; i++) {
            scl->max[i] = scl->min[i] = x[0][i];
            for (j = 1; j < samples; j++) {
                if (scl->max[i] < x[j][i]) scl->max[i] = x[j][i];
                if (scl->min[i] > x[j][i]) scl->min[i] = x[j][i];
            }
        }
        if (y) {
            scl->max[scl->features - 1] = scl->min[scl->features - 1] = y[0];
            for (i = 1; i < samples; i++) {
                if (scl->max[scl->features - 1] < y[i]) scl->max[scl->features - 1] = y[i];
                if (scl->min[scl->features - 1] > y[i]) scl->min[scl->features - 1] = y[i];
            }
        }
    }
}
void scaler_transform(float** x, float* y, int samples, int features, void* scaler, char* scaler_type) {
    if (!strcmp(scaler_type, "Standard_scaler")) {
        Standard_scaler* scl = (Standard_scaler*)scaler;
        int i, j;
        for (i = 0; i < features; i++) {
            for (j = 0; j < samples; j++) x[j][i] = (x[j][i] - scl->mean[i]) / scl->deviation[i];
        }
        if (y)
            for (i = 0; i < samples; i++) 
                y[i] = (y[i] - scl->mean[scl->features - 1]) / scl->deviation[scl->features - 1];;
    } else if (!strcmp(scaler_type, "Min_max_scaler")) {
        Min_max_scaler* scl = (Min_max_scaler*)scaler;
        int i, j;
        for (i = 0; i < features; i++) {
            for (j = 0; j < samples; j++) x[j][i] = (x[j][i] - scl->min[i]) / (scl->max[i] - scl->min[i]);
        }
        if (y)
            for (i = 0; i < samples; i++) 
                y[i] = (y[i] - scl->min[scl->features - 1]) / (scl->max[scl->features - 1] - scl->min[scl->features - 1]);;
    }
}
void free_scaler(void* scaler, char* scaler_type) {
    if (!strcmp(scaler_type, "Standard_scaler")) {
        Standard_scaler* scl = (Standard_scaler*)scaler;
        free(scl->mean);
        free(scl->deviation);
        free(scl);
    } else if (!strcmp(scaler_type, "Min_max_scaler")) {
        Min_max_scaler* scl = (Min_max_scaler*)scaler;
        free(scl->max);
        free(scl->min);
        free(scl);
    }
}
void shuffle_index(int* index, int size, int random_state) {
	srand(random_state ^ time(NULL) ^ clock());
    size *= 2; size /= 3;
	for (int j = 0, t, a, b; j < size; j++) {
		a = rand() % size, b = rand() % size;
		t = index[a];
		index[a] = index[b];
		index[b] = t;
	}
}
void encoder_fit(char*** data, int num_samples, int col, void* encoder, char* encoder_type) {
    if (!strcmp(encoder_type, "One_hot_encoder")) {
        One_hot_encoder* encode = (One_hot_encoder*) encoder;
        int i, j = 0;
        encode->sample_types = new_set(17);
        for (i = 0; i < num_samples; i++) 
            if (!set_find(encode->sample_types, data[i][col])) set_add(encode->sample_types, data[i][col], j++);;
    } else if (!strcmp(encoder_type, "Label_encoder")) {
        Label_encoder* encode = (Label_encoder*) encoder;
        int i, j = 0;
        encode->sample_types = new_set(17);
        for (i = 0; i < num_samples; i++) 
            if (!set_find(encode->sample_types, data[i][col])) set_add(encode->sample_types, data[i][col], j++);;
    }
}
void* encoder_transform(char*** data, int num_samples, int col, void* encoder, char* encoder_type) {
    if (!strcmp(encoder_type, "One_hot_encoder")) {
        One_hot_encoder* encode = (One_hot_encoder*) encoder;
        float** encd = (float**)malloc(num_samples* sizeof(float*));
        for (int i = 0, j; i < num_samples; i++) {
            j = set_call(encode->sample_types, data[i][col]);
            encd[i] = (float*)calloc(encode->sample_types->size, sizeof(float));
            if (j != -1) encd[i][j] = 1;
        }
        return (void*)encd;
    } else if (!strcmp(encoder_type, "Label_encoder")) {
        Label_encoder* encode = (Label_encoder*) encoder;
        float* encd = (float*)malloc(num_samples* sizeof(float));
        for (int i = 0; i < num_samples; i++)
            encd[i] = (float) set_call(encode->sample_types, data[i][col]);
        return (void*)encd;
    }
}
float** label_to_one_hot_encode(float** data, int col, float* line_data, int samples, int* class_num) {
    int i, j = 0;
    if (!line_data) {
        if (!data) return NULL;
        line_data = (float*)calloc(samples, sizeof(float));
        for (i = 0; i < samples; i++) line_data[i] = data[i][col];
    }
    Set* st = new_set(17);
    char keys[10];
    for (i = 0; i < samples; i++) {
        ftostr(line_data[i], keys, 3);
        if (!set_find(st, keys)) set_add(st, keys, j++);
    }
    *class_num = st->size;
    float** ohe = new_matrix(samples, *class_num);
    for (i = 0; i < samples; i++) {
        ftostr(line_data[i], keys, 3);
        j = set_call(st, keys);
        if (j != -1) ohe[i][j] = 1.0;
    }
    free_set(st);
    if (data) free(line_data);
    return ohe;
}
void free_encoder(void* encoder, char* encoder_type) {
    if (!strcmp(encoder_type, "One_hot_encoder")) {
        One_hot_encoder* encode = (One_hot_encoder*) encoder;
        free_set(encode->sample_types);
        free(encode);
    } else if (!strcmp(encoder_type, "Label_encoder")) {
        Label_encoder* encode = (Label_encoder*) encoder;
        free_set(encode->sample_types);
        free(encode);
    }
}
Simple_Imputer* simple_impute(Data_Frame* df, char* strategy, float* fill_value, char** fill_str_value) {
    Simple_Imputer* news = (Simple_Imputer*)malloc(sizeof(Simple_Imputer));
    int i, j, k;
    news->digit_data = (float*)calloc(df->col, sizeof(float));
    if (!strcmp(strategy, "mean")) {
        for (j = 0; j < df->col; j++) {
            for (i = 0, k = 0; i < df->row; i++) {
                if (df->data[i][j] == df->data[i][j]) {
                    news->digit_data[j] += df->data[i][j];
                    k++;
                }
            }
            news->digit_data[j] /= k;
        }
    } else if (!strcmp(strategy, "median")) {
        float* cur;
        for (j = 0; j < df->col; j++, k = 0) {
            for (i = 0; i < df->row; i++) if (df->data[i][j] == df->data[i][j]) k++;
            cur = (float*)malloc(k* sizeof(float));
            for (i = 0, k = 0; i < df->row; i++, k++)
                if (df->data[i][j] == df->data[i][j]) cur[k] = df->data[i][j];
            news->digit_data[j] = median(cur, k);
            free(cur);
        }
    } else if (!strcmp(strategy, "constant")) for (i = 0; i < df->col; i++) news->digit_data[i] = fill_value[i];
    if (df->str_cols[0] != 0 && fill_str_value) {
        news->str_data = (char**)malloc((df->str_cols[0] + 1)* sizeof(char*));
        if (!strcmp(fill_str_value[0], "most_frequent")) {
            Set** str_frequence = (Set**)malloc(df->str_cols[0]* sizeof(Set*));
            for (i = 0; i < df->str_cols[0]; i++) {
                str_frequence[i] = new_set(17);
                for (j = 0; j < df->row; j++) {
                    if (strcmp(df->str_data[j][i], "nan")) {
                        if (set_find(str_frequence[i], df->str_data[j][i]))
                            (*set_key_access(str_frequence[i], df->str_data[j][i]))++;
                        else set_add(str_frequence[i], df->str_data[j][i], 1);
                    }
                }
            }
            Space* mid;
            for (k = 0; k < df->str_cols[0]; k++) {
                for (i = 0, j = 0; i < str_frequence[k]->max_size; i++) {
                    mid = str_frequence[k]->arr[i];
                    while (mid != NULL) {
                        if (mid->key > j) {
                            j = strlen(mid->keys) + 1;
                            news->str_data[k] = (char*)malloc(j* sizeof(char));
                            strcpy(news->str_data[k], mid->keys);
                            j = mid->key;
                        }
                        mid = mid->next;
                    }
                }
                free_set(str_frequence[k]);
            }
            free(str_frequence);
        } else {
            for (i = 0; i < df->str_cols[0]; i++) {
                if (fill_str_value[i]) {
                    news->str_data[i] = (char*)malloc((strlen(fill_str_value[i]) + 1)* sizeof(char));
                    strcpy(news->str_data[i], fill_str_value[i]);
                }
            }
        }
        news->str_data[df->str_cols[0]] = (char*)malloc(5* sizeof(char));
        strcpy(news->str_data[df->str_cols[0]], "null");
    }
    return news;
}
void simple_impute_transform(Data_Frame* df, Simple_Imputer* imputer) {
    int i, j;
    for (i = 0; i < df->col; i++) {
        for (j = 0; j < df->row; j++)
            if (df->data[j][i] != df->data[j][i]) df->data[j][i] = imputer->digit_data[i];
    }
    if (imputer->str_data) {
        for (i = 0; i < df->str_cols[0]; i++) {
            if (imputer->str_data[i]) {
                for (j = 0; j < df->row; j++)
                    if (!strcmp(df->str_data[j][i], "nan"))
                        strcpy(df->str_data[j][i], imputer->str_data[i]);
            }
        }
    }
}
void free_simple_imputer(Simple_Imputer* imputer) {
    if (imputer->digit_data) free(imputer->digit_data);
    if (imputer->str_data) {
        int i;
        for (i = 0; strcmp(imputer->str_data[i], "null"); i++) free(imputer->str_data[i]);
        free(imputer->str_data[i]);
        free(imputer->str_data);
    }
    free(imputer);
}
/*void train_test_split_ds(Dataset* data, Dataset* train, Dataset* test, float test_size, int random_state) {
    train->features = data->features;
    test->features = data->features;
    test->samples = round(test_size* data->samples);
    train->samples = data->samples - test->samples;
    test->x = (float**)malloc(test->samples* sizeof(float*));
    test->y = (float*)malloc(test->samples* sizeof(float));
    train->x = (float**)malloc(train->samples* sizeof(float*));
    train->y = (float*)malloc(train->samples* sizeof(float));

    int* random_i = (int*)malloc(data->samples* sizeof(int)), i;
    for (i = 0; i < data->samples; i++) random_i[i] = i;
    shuffle_index(random_i, data->samples, random_state);
    
    for (i = 0; i < test->samples; i++) {
        test->x[i] = (float*)malloc((test->features + 1)* sizeof(float));
        dataset_sample_copy(data, random_i[i], test, i);
    }
    for (int e = 0 ; i < data->samples; i++, e++) {
        train->x[e] = (float*)malloc((train->features + 1)* sizeof(float));
        dataset_sample_copy(data, random_i[i], train, e);
    }
    free(random_i);
}*/
