static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t  i;
    CUMO_BIT_DIGIT *a;
    size_t  p1, p2;
    ssize_t s1, s2;
    size_t *idx1, *idx2, *pidx;
    CUMO_BIT_DIGIT x=0;
    size_t  count;
    where_opt_t *g;

    CUMO_SHOW_SYNCHRONIZE_WARNING_ONCE("<%=name%>", "<%=type_name%>");
    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());

    g = (where_opt_t*)(lp->opt_ptr);
    count = g->count;
    pidx  = (size_t*)(g->idx1);
    CUMO_INIT_COUNTER(lp, i);
    CUMO_INIT_PTR_BIT_IDX(lp, 0, a, p1, s1, idx1);
    //CUMO_INIT_PTR_IDX(lp, 1, p2, s2, idx2);
    p2 = lp->args[1].iter[0].pos;
    s2 = lp->args[1].iter[0].step;
    idx2 = lp->args[1].iter[0].idx;

    if (idx1) {
        if (idx2) {
            for (; i--;) {
                CUMO_LOAD_BIT(a, p1+*idx1, x);
                idx1++;
                if (x) {
                    *(pidx++) = p2+*idx2;
                    count++;
                }
                idx2++;
            }
        } else {
            for (; i--;) {
                CUMO_LOAD_BIT(a, p1+*idx1, x);
                idx1++;
                if (x) {
                    *(pidx++) = p2;
                    count++;
                }
                p2 += s2;
            }
        }
    } else {
        if (idx2) {
            for (; i--;) {
                CUMO_LOAD_BIT(a, p1, x);
                p1 += s1;
                if (x) {
                    *(pidx++) = p2+*idx2;
                    count++;
                }
                idx2++;
            }
        } else {
            for (; i--;) {
                CUMO_LOAD_BIT(a, p1, x);
                p1 += s1;
                if (x) {
                    *(pidx++) = p2;
                    count++;
                }
                p2 += s2;
            }
        }
    }
    g->count = count;
    g->idx1  = (char*)pidx;
}

#if   SIZEOF_VOIDP == 8
#define cIndex cumo_cInt64
#elif SIZEOF_VOIDP == 4
#define cIndex cumo_cInt32
#endif

static void shape_error(void) {
    rb_raise(cumo_na_eShapeError,"mask and masked arrays must have the same shape");
}

/*
  Return subarray of argument masked with self bit array.
  @overload <%=op_map%>(array)
  @param [Cumo::NArray] array  narray to be masked.
  @return [Cumo::NArray]  view of masked array.
*/
static VALUE
<%=c_func(1)%>(VALUE mask, VALUE val)
{
    int i;
    VALUE idx_1, view;
    cumo_narray_data_t *nidx;
    cumo_narray_view_t *nv, *nv_val;
    cumo_narray_t      *na, *na_mask;
    cumo_stridx_t stridx0;
    size_t n_1;
    where_opt_t g;
    cumo_ndfunc_arg_in_t ain[2] = {{cT,0},{Qnil,0}};
    cumo_ndfunc_t ndf = {<%=c_iter%>, CUMO_FULL_LOOP, 2, 0, ain, 0};

    // cast val to NArray
    if (!rb_obj_is_kind_of(val, cumo_cNArray)) {
        val = rb_funcall(cumo_cNArray, cumo_id_cast, 1, val);
    }
    // shapes of mask and val must be same
    CumoGetNArray(val, na);
    CumoGetNArray(mask, na_mask);
    if (na_mask->ndim != na->ndim) {
        shape_error();
    }
    for (i=0; i<na->ndim; i++) {
        if (na_mask->shape[i] != na->shape[i]) {
            shape_error();
        }
    }

    // TODO(sonots): bit_count_true synchronizes with CPU. Avoid.
    n_1 = NUM2SIZET(<%=find_tmpl("count_true_cpu").c_func%>(0, NULL, mask));
    idx_1 = cumo_na_new(cIndex, 1, &n_1);
    g.count = 0;
    g.elmsz = SIZEOF_VOIDP;
    g.idx1 = cumo_na_get_pointer_for_write(idx_1);
    g.idx0 = NULL;
    cumo_na_ndloop3(&ndf, &g, 2, mask, val);

    view = cumo_na_s_allocate_view(rb_obj_class(val));
    CumoGetNArrayView(view, nv);
    cumo_na_setup_shape((cumo_narray_t*)nv, 1, &n_1);

    CumoGetNArrayData(idx_1,nidx);
    CUMO_SDX_SET_INDEX(stridx0,(size_t*)nidx->ptr);
    nidx->ptr = NULL;
    RB_GC_GUARD(idx_1);

    nv->stridx = ALLOC_N(cumo_stridx_t,1);
    nv->stridx[0] = stridx0;
    nv->offset = 0;

    switch(CUMO_NA_TYPE(na)) {
    case CUMO_NARRAY_DATA_T:
        nv->data = val;
        break;
    case CUMO_NARRAY_VIEW_T:
        CumoGetNArrayView(val, nv_val);
        nv->data = nv_val->data;
        break;
    default:
        rb_raise(rb_eRuntimeError,"invalid CUMO_NA_TYPE: %d",CUMO_NA_TYPE(na));
    }

    return view;
}
