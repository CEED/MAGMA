///////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _UNPACK_ARGS_
#define _UNPACK_ARGS_

#define quark_unpack_args_1(quark, \
    arg1) \
{ \
  icl_list_t *args_list = QUARK_Args_List( quark ); \
  icl_list_t *arg_node = icl_list_first(args_list); \
  memcpy(&arg1, arg_node->data, sizeof(arg1)); \
}

#define quark_unpack_args_2(quark, \
    arg1, \
    arg2) \
{ \
  icl_list_t *args_list = QUARK_Args_List( quark ); \
  icl_list_t *arg_node = icl_list_first(args_list); \
  memcpy(&arg1, arg_node->data, sizeof(arg1)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg2, arg_node->data, sizeof(arg2)); \
}

#define quark_unpack_args_3(quark, \
    arg1, \
    arg2, \
    arg3) \
{ \
  icl_list_t *args_list = QUARK_Args_List( quark ); \
  icl_list_t *arg_node = icl_list_first(args_list); \
  memcpy(&arg1, arg_node->data, sizeof(arg1)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg2, arg_node->data, sizeof(arg2)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg3, arg_node->data, sizeof(arg3)); \
}

#define quark_unpack_args_4(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4) \
{ \
  icl_list_t *args_list = QUARK_Args_List( quark ); \
  icl_list_t *arg_node = icl_list_first(args_list); \
  memcpy(&arg1, arg_node->data, sizeof(arg1)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg2, arg_node->data, sizeof(arg2)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg3, arg_node->data, sizeof(arg3)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg4, arg_node->data, sizeof(arg4)); \
}

#define quark_unpack_args_5(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5) \
{ \
  icl_list_t *args_list = QUARK_Args_List( quark ); \
  icl_list_t *arg_node = icl_list_first(args_list); \
  memcpy(&arg1, arg_node->data, sizeof(arg1)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg2, arg_node->data, sizeof(arg2)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg3, arg_node->data, sizeof(arg3)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg4, arg_node->data, sizeof(arg4)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg5, arg_node->data, sizeof(arg5)); \
}

#define quark_unpack_args_6(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6) \
{ \
  icl_list_t *args_list = QUARK_Args_List( quark ); \
  icl_list_t *arg_node = icl_list_first(args_list); \
  memcpy(&arg1, arg_node->data, sizeof(arg1)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg2, arg_node->data, sizeof(arg2)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg3, arg_node->data, sizeof(arg3)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg4, arg_node->data, sizeof(arg4)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg5, arg_node->data, sizeof(arg5)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg6, arg_node->data, sizeof(arg6)); \
}

#define quark_unpack_args_7(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7) \
{ \
  icl_list_t *args_list = QUARK_Args_List( quark ); \
  icl_list_t *arg_node = icl_list_first(args_list); \
  memcpy(&arg1, arg_node->data, sizeof(arg1)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg2, arg_node->data, sizeof(arg2)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg3, arg_node->data, sizeof(arg3)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg4, arg_node->data, sizeof(arg4)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg5, arg_node->data, sizeof(arg5)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg6, arg_node->data, sizeof(arg6)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg7, arg_node->data, sizeof(arg7)); \
}

#define quark_unpack_args_8(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8) \
{ \
  icl_list_t *args_list = QUARK_Args_List( quark ); \
  icl_list_t *arg_node = icl_list_first(args_list); \
  memcpy(&arg1, arg_node->data, sizeof(arg1)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg2, arg_node->data, sizeof(arg2)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg3, arg_node->data, sizeof(arg3)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg4, arg_node->data, sizeof(arg4)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg5, arg_node->data, sizeof(arg5)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg6, arg_node->data, sizeof(arg6)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg7, arg_node->data, sizeof(arg7)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg8, arg_node->data, sizeof(arg8)); \
}

#define quark_unpack_args_9(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9) \
{ \
  icl_list_t *args_list = QUARK_Args_List( quark ); \
  icl_list_t *arg_node = icl_list_first(args_list); \
  memcpy(&arg1, arg_node->data, sizeof(arg1)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg2, arg_node->data, sizeof(arg2)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg3, arg_node->data, sizeof(arg3)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg4, arg_node->data, sizeof(arg4)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg5, arg_node->data, sizeof(arg5)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg6, arg_node->data, sizeof(arg6)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg7, arg_node->data, sizeof(arg7)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg8, arg_node->data, sizeof(arg8)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg9, arg_node->data, sizeof(arg9)); \
}

#define quark_unpack_args_10(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10) \
{ \
  icl_list_t *args_list = QUARK_Args_List( quark ); \
  icl_list_t *arg_node = icl_list_first(args_list); \
  memcpy(&arg1, arg_node->data, sizeof(arg1)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg2, arg_node->data, sizeof(arg2)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg3, arg_node->data, sizeof(arg3)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg4, arg_node->data, sizeof(arg4)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg5, arg_node->data, sizeof(arg5)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg6, arg_node->data, sizeof(arg6)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg7, arg_node->data, sizeof(arg7)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg8, arg_node->data, sizeof(arg8)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg9, arg_node->data, sizeof(arg9)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg10, arg_node->data, sizeof(arg10)); \
}

#define quark_unpack_args_11(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10, \
    arg11) \
{ \
  icl_list_t *args_list = QUARK_Args_List( quark ); \
  icl_list_t *arg_node = icl_list_first(args_list); \
  memcpy(&arg1, arg_node->data, sizeof(arg1)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg2, arg_node->data, sizeof(arg2)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg3, arg_node->data, sizeof(arg3)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg4, arg_node->data, sizeof(arg4)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg5, arg_node->data, sizeof(arg5)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg6, arg_node->data, sizeof(arg6)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg7, arg_node->data, sizeof(arg7)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg8, arg_node->data, sizeof(arg8)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg9, arg_node->data, sizeof(arg9)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg10, arg_node->data, sizeof(arg10)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg11, arg_node->data, sizeof(arg11)); \
}

#define quark_unpack_args_12(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10, \
    arg11, \
    arg12) \
{ \
  icl_list_t *args_list = QUARK_Args_List( quark ); \
  icl_list_t *arg_node = icl_list_first(args_list); \
  memcpy(&arg1, arg_node->data, sizeof(arg1)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg2, arg_node->data, sizeof(arg2)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg3, arg_node->data, sizeof(arg3)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg4, arg_node->data, sizeof(arg4)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg5, arg_node->data, sizeof(arg5)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg6, arg_node->data, sizeof(arg6)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg7, arg_node->data, sizeof(arg7)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg8, arg_node->data, sizeof(arg8)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg9, arg_node->data, sizeof(arg9)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg10, arg_node->data, sizeof(arg10)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg11, arg_node->data, sizeof(arg11)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg12, arg_node->data, sizeof(arg12)); \
}

#define quark_unpack_args_13(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10, \
    arg11, \
    arg12, \
    arg13) \
{ \
  icl_list_t *args_list = QUARK_Args_List( quark ); \
  icl_list_t *arg_node = icl_list_first(args_list); \
  memcpy(&arg1, arg_node->data, sizeof(arg1)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg2, arg_node->data, sizeof(arg2)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg3, arg_node->data, sizeof(arg3)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg4, arg_node->data, sizeof(arg4)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg5, arg_node->data, sizeof(arg5)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg6, arg_node->data, sizeof(arg6)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg7, arg_node->data, sizeof(arg7)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg8, arg_node->data, sizeof(arg8)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg9, arg_node->data, sizeof(arg9)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg10, arg_node->data, sizeof(arg10)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg11, arg_node->data, sizeof(arg11)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg12, arg_node->data, sizeof(arg12)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg13, arg_node->data, sizeof(arg13)); \
}

#define quark_unpack_args_14(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10, \
    arg11, \
    arg12, \
    arg13, \
    arg14) \
{ \
  icl_list_t *args_list = QUARK_Args_List( quark ); \
  icl_list_t *arg_node = icl_list_first(args_list); \
  memcpy(&arg1, arg_node->data, sizeof(arg1)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg2, arg_node->data, sizeof(arg2)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg3, arg_node->data, sizeof(arg3)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg4, arg_node->data, sizeof(arg4)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg5, arg_node->data, sizeof(arg5)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg6, arg_node->data, sizeof(arg6)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg7, arg_node->data, sizeof(arg7)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg8, arg_node->data, sizeof(arg8)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg9, arg_node->data, sizeof(arg9)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg10, arg_node->data, sizeof(arg10)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg11, arg_node->data, sizeof(arg11)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg12, arg_node->data, sizeof(arg12)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg13, arg_node->data, sizeof(arg13)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg14, arg_node->data, sizeof(arg14)); \
}

#define quark_unpack_args_15(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10, \
    arg11, \
    arg12, \
    arg13, \
    arg14, \
    arg15) \
{ \
  icl_list_t *args_list = QUARK_Args_List( quark ); \
  icl_list_t *arg_node = icl_list_first(args_list); \
  memcpy(&arg1, arg_node->data, sizeof(arg1)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg2, arg_node->data, sizeof(arg2)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg3, arg_node->data, sizeof(arg3)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg4, arg_node->data, sizeof(arg4)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg5, arg_node->data, sizeof(arg5)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg6, arg_node->data, sizeof(arg6)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg7, arg_node->data, sizeof(arg7)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg8, arg_node->data, sizeof(arg8)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg9, arg_node->data, sizeof(arg9)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg10, arg_node->data, sizeof(arg10)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg11, arg_node->data, sizeof(arg11)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg12, arg_node->data, sizeof(arg12)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg13, arg_node->data, sizeof(arg13)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg14, arg_node->data, sizeof(arg14)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg15, arg_node->data, sizeof(arg15)); \
}

#define quark_unpack_args_16(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10, \
    arg11, \
    arg12, \
    arg13, \
    arg14, \
    arg15, \
    arg16) \
{ \
  icl_list_t *args_list = QUARK_Args_List( quark ); \
  icl_list_t *arg_node = icl_list_first(args_list); \
  memcpy(&arg1, arg_node->data, sizeof(arg1)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg2, arg_node->data, sizeof(arg2)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg3, arg_node->data, sizeof(arg3)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg4, arg_node->data, sizeof(arg4)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg5, arg_node->data, sizeof(arg5)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg6, arg_node->data, sizeof(arg6)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg7, arg_node->data, sizeof(arg7)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg8, arg_node->data, sizeof(arg8)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg9, arg_node->data, sizeof(arg9)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg10, arg_node->data, sizeof(arg10)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg11, arg_node->data, sizeof(arg11)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg12, arg_node->data, sizeof(arg12)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg13, arg_node->data, sizeof(arg13)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg14, arg_node->data, sizeof(arg14)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg15, arg_node->data, sizeof(arg15)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg16, arg_node->data, sizeof(arg16)); \
}

#define quark_unpack_args_17(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10, \
    arg11, \
    arg12, \
    arg13, \
    arg14, \
    arg15, \
    arg16, \
    arg17) \
{ \
  icl_list_t *args_list = QUARK_Args_List( quark ); \
  icl_list_t *arg_node = icl_list_first(args_list); \
  memcpy(&arg1, arg_node->data, sizeof(arg1)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg2, arg_node->data, sizeof(arg2)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg3, arg_node->data, sizeof(arg3)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg4, arg_node->data, sizeof(arg4)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg5, arg_node->data, sizeof(arg5)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg6, arg_node->data, sizeof(arg6)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg7, arg_node->data, sizeof(arg7)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg8, arg_node->data, sizeof(arg8)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg9, arg_node->data, sizeof(arg9)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg10, arg_node->data, sizeof(arg10)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg11, arg_node->data, sizeof(arg11)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg12, arg_node->data, sizeof(arg12)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg13, arg_node->data, sizeof(arg13)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg14, arg_node->data, sizeof(arg14)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg15, arg_node->data, sizeof(arg15)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg16, arg_node->data, sizeof(arg16)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg17, arg_node->data, sizeof(arg17)); \
}

#define quark_unpack_args_18(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10, \
    arg11, \
    arg12, \
    arg13, \
    arg14, \
    arg15, \
    arg16, \
    arg17, \
    arg18) \
{ \
  icl_list_t *args_list = QUARK_Args_List( quark ); \
  icl_list_t *arg_node = icl_list_first(args_list); \
  memcpy(&arg1, arg_node->data, sizeof(arg1)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg2, arg_node->data, sizeof(arg2)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg3, arg_node->data, sizeof(arg3)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg4, arg_node->data, sizeof(arg4)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg5, arg_node->data, sizeof(arg5)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg6, arg_node->data, sizeof(arg6)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg7, arg_node->data, sizeof(arg7)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg8, arg_node->data, sizeof(arg8)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg9, arg_node->data, sizeof(arg9)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg10, arg_node->data, sizeof(arg10)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg11, arg_node->data, sizeof(arg11)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg12, arg_node->data, sizeof(arg12)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg13, arg_node->data, sizeof(arg13)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg14, arg_node->data, sizeof(arg14)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg15, arg_node->data, sizeof(arg15)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg16, arg_node->data, sizeof(arg16)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg17, arg_node->data, sizeof(arg17)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg18, arg_node->data, sizeof(arg18)); \
}

#define quark_unpack_args_19(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10, \
    arg11, \
    arg12, \
    arg13, \
    arg14, \
    arg15, \
    arg16, \
    arg17, \
    arg18, \
    arg19) \
{ \
  icl_list_t *args_list = QUARK_Args_List( quark ); \
  icl_list_t *arg_node = icl_list_first(args_list); \
  memcpy(&arg1, arg_node->data, sizeof(arg1)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg2, arg_node->data, sizeof(arg2)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg3, arg_node->data, sizeof(arg3)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg4, arg_node->data, sizeof(arg4)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg5, arg_node->data, sizeof(arg5)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg6, arg_node->data, sizeof(arg6)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg7, arg_node->data, sizeof(arg7)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg8, arg_node->data, sizeof(arg8)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg9, arg_node->data, sizeof(arg9)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg10, arg_node->data, sizeof(arg10)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg11, arg_node->data, sizeof(arg11)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg12, arg_node->data, sizeof(arg12)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg13, arg_node->data, sizeof(arg13)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg14, arg_node->data, sizeof(arg14)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg15, arg_node->data, sizeof(arg15)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg16, arg_node->data, sizeof(arg16)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg17, arg_node->data, sizeof(arg17)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg18, arg_node->data, sizeof(arg18)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg19, arg_node->data, sizeof(arg19)); \
}

#define quark_unpack_args_20(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10, \
    arg11, \
    arg12, \
    arg13, \
    arg14, \
    arg15, \
    arg16, \
    arg17, \
    arg18, \
    arg19, \
    arg20) \
{ \
  icl_list_t *args_list = QUARK_Args_List( quark ); \
  icl_list_t *arg_node = icl_list_first(args_list); \
  memcpy(&arg1, arg_node->data, sizeof(arg1)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg2, arg_node->data, sizeof(arg2)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg3, arg_node->data, sizeof(arg3)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg4, arg_node->data, sizeof(arg4)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg5, arg_node->data, sizeof(arg5)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg6, arg_node->data, sizeof(arg6)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg7, arg_node->data, sizeof(arg7)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg8, arg_node->data, sizeof(arg8)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg9, arg_node->data, sizeof(arg9)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg10, arg_node->data, sizeof(arg10)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg11, arg_node->data, sizeof(arg11)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg12, arg_node->data, sizeof(arg12)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg13, arg_node->data, sizeof(arg13)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg14, arg_node->data, sizeof(arg14)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg15, arg_node->data, sizeof(arg15)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg16, arg_node->data, sizeof(arg16)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg17, arg_node->data, sizeof(arg17)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg18, arg_node->data, sizeof(arg18)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg19, arg_node->data, sizeof(arg19)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg20, arg_node->data, sizeof(arg20)); \
}

#define quark_unpack_args_21(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10, \
    arg11, \
    arg12, \
    arg13, \
    arg14, \
    arg15, \
    arg16, \
    arg17, \
    arg18, \
    arg19, \
    arg20, \
    arg21) \
{ \
  icl_list_t *args_list = QUARK_Args_List( quark ); \
  icl_list_t *arg_node = icl_list_first(args_list); \
  memcpy(&arg1, arg_node->data, sizeof(arg1)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg2, arg_node->data, sizeof(arg2)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg3, arg_node->data, sizeof(arg3)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg4, arg_node->data, sizeof(arg4)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg5, arg_node->data, sizeof(arg5)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg6, arg_node->data, sizeof(arg6)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg7, arg_node->data, sizeof(arg7)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg8, arg_node->data, sizeof(arg8)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg9, arg_node->data, sizeof(arg9)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg10, arg_node->data, sizeof(arg10)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg11, arg_node->data, sizeof(arg11)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg12, arg_node->data, sizeof(arg12)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg13, arg_node->data, sizeof(arg13)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg14, arg_node->data, sizeof(arg14)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg15, arg_node->data, sizeof(arg15)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg16, arg_node->data, sizeof(arg16)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg17, arg_node->data, sizeof(arg17)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg18, arg_node->data, sizeof(arg18)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg19, arg_node->data, sizeof(arg19)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg20, arg_node->data, sizeof(arg20)); arg_node = icl_list_next(args_list, arg_node); \
  memcpy(&arg21, arg_node->data, sizeof(arg21)); \
}



#endif
