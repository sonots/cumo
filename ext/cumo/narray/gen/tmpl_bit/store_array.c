void <%="cumo_#{c_iter}_index_kernel_launch"%>(CUMO_BIT_DIGIT *a1, size_t p1, size_t *idx1, CUMO_BIT_DIGIT* z, uint64_t n);
void <%="cumo_#{c_iter}_stride_kernel_launch"%>(CUMO_BIT_DIGIT *a1, size_t p1, ssize_t s1, CUMO_BIT_DIGIT* z, uint64_t n);
void <%="cumo_#{c_iter}_contiguous_kernel_launch"%>(CUMO_BIT_DIGIT *a1, size_t p1, CUMO_BIT_DIGIT* z, uint64_t n);
void <%="cumo_#{c_iter}_index_scalar_kernel_launch"%>(CUMO_BIT_DIGIT *a1, size_t p1, size_t *idx1, CUMO_BIT_DIGIT z, uint64_t n);
void <%="cumo_#{c_iter}_stride_scalar_kernel_launch"%>(CUMO_BIT_DIGIT *a1, size_t p1, ssize_t s1, CUMO_BIT_DIGIT z, uint64_t n);

#define <%=c_iter%>_set(buf, i, z)                                      \
    if ((z) & 1) { (buf)[(i) / CUMO_NB] |= (CUMO_BIT_DIGIT)1 << ((i) % CUMO_NB); }

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t i, n;
    size_t i1, n1;
    VALUE  v1;
    CUMO_BIT_DIGIT *a1;
    size_t p1;
    // Every other Bit iterator declares the step signed, which is what tells
    // CUMO_INIT_PTR_BIT_IDX to leave the position alone for a view that walks
    // backwards. Unsigned here, it normalized the pointer to the first word of
    // the view and then walked below it.
    ssize_t s1;
    size_t *idx1;
    VALUE  x;
    double y;
    CUMO_BIT_DIGIT z;
    size_t len, c;
    double beg, step;

    CUMO_INIT_COUNTER(lp, n);
    CUMO_INIT_PTR_BIT_IDX(lp, 0, a1, p1, s1, idx1);
    v1 = lp->args[1].value;
    i = 0;

    if (lp->args[1].ptr) {
        if (v1 == Qtrue) {
            // The loop counter is the destination length, but the sub-narray
            // may be shorter. Copy only what the source actually holds, or the
            // kernel would read past its end; the rest is zero-filled below.
            i = lp->args[1].shape[0];
            if (i > n) {
                i = n;
            }
            CUMO_NDL_CNT(lp) = i;
            iter_<%=type_name%>_store_<%=type_name%>(lp);
            CUMO_NDL_CNT(lp) = n;
        }
        goto loop_end;
    }

    switch(TYPE(v1)) {
    case T_ARRAY:
        n1 = RARRAY_LEN(v1);
        break;
    case T_NIL:
        n1 = 0;
        break;
    default:
        n1 = 1;
    }

    {
        // Reading a Ruby Array is host work whatever happens, so the bits are
        // packed into a staging buffer here and one copy carries them over.
        //
        // The staging buffer is held by a Ruby object rather than a plain local:
        // the conversion loop below raises for a value which is not a number,
        // and ndloop has no way to free a buffer it does not know about.
        VALUE  buf;
        uint64_t nw = (n + CUMO_NB - 1) / CUMO_NB;
        CUMO_BIT_DIGIT* host_z = RB_ALLOCV_N(CUMO_BIT_DIGIT, buf, nw ? nw : 1);

        memset(host_z, 0, sizeof(CUMO_BIT_DIGIT) * (nw ? nw : 1));
        for (i=i1=0; i1<n1 && i<n; i1++) {
            if (!cumo_na_store_rary_fetch(v1, i1, &x)) break;
#ifdef HAVE_RB_ARITHMETIC_SEQUENCE_EXTRACT
            if (rb_obj_is_kind_of(x, rb_cRange) || rb_obj_is_kind_of(x, rb_cArithSeq)) {
#else
            if (rb_obj_is_kind_of(x, rb_cRange) || rb_obj_is_kind_of(x, rb_cEnumerator)) {
#endif
                cumo_na_step_sequence(x,&len,&beg,&step);
                for (c=0; c<len && i<n; c++,i++) {
                    y = beg + step * c;
                    z = m_from_double(y);
                    <%=c_iter%>_set(host_z, i, z);
                }
            }
            else if (TYPE(x) != T_ARRAY) {
                if (x == Qnil) x = INT2FIX(0);
                z = m_num_to_data(x);
                <%=c_iter%>_set(host_z, i, z);
                i++;
            }
        }

        // A run that starts on a word and ends on one owns every word it
        // covers, so it goes straight over with no staging buffer of its own.
        if (!idx1 && s1 == 1 && p1 == 0 && i % CUMO_NB == 0) {
            cumo_cuda_runtime_check_status(
                cudaMemcpyAsync(a1,host_z,sizeof(CUMO_BIT_DIGIT)*(i/CUMO_NB),cudaMemcpyHostToDevice,0));
        } else {
            uint64_t iw = (i + CUMO_NB - 1) / CUMO_NB;
            CUMO_BIT_DIGIT* device_z =
                (CUMO_BIT_DIGIT*)cumo_cuda_runtime_malloc(sizeof(CUMO_BIT_DIGIT) * (iw ? iw : 1));
            cudaError_t status =
                cudaMemcpyAsync(device_z,host_z,sizeof(CUMO_BIT_DIGIT)*iw,cudaMemcpyHostToDevice,0);
            if (status == 0 && i > 0) {
                if (idx1) {
                    <%="cumo_#{c_iter}_index_kernel_launch"%>(a1,p1,idx1,device_z,i);
                } else if (s1 == 1) {
                    <%="cumo_#{c_iter}_contiguous_kernel_launch"%>(a1,p1,device_z,i);
                } else {
                    <%="cumo_#{c_iter}_stride_kernel_launch"%>(a1,p1,s1,device_z,i);
                }
                // The kernel reads device_z, so wait for it before giving the
                // buffer back. The memory pool hands a freed chunk straight out
                // again, and a host write into that managed memory does not wait
                // for the stream, so the kernel would read the new contents.
                status = cudaStreamSynchronize(0);
            }
            cumo_cuda_runtime_free((void*)device_z);
            cumo_cuda_runtime_check_status(status);
        }
        RB_ALLOCV_END(buf);
    }

 loop_end:
    z = m_zero;
    // i may exceed n when the sub-narray is longer than the destination;
    // n-i would then wrap around as size_t.
    if (i < n) {
        if (idx1) {
            <%="cumo_#{c_iter}_index_scalar_kernel_launch"%>(a1,p1,idx1+i,z,n-i);
        } else {
            <%="cumo_#{c_iter}_stride_scalar_kernel_launch"%>(a1,p1+s1*i,s1,z,n-i);
        }
    }
}

static VALUE
<%=c_func(:nodef)%>(VALUE self, VALUE rary)
{
    cumo_ndfunc_arg_in_t ain[2] = {{CUMO_OVERWRITE,0}, {rb_cArray,0}};
    cumo_ndfunc_t ndf = {<%=c_iter%>, CUMO_FULL_LOOP, 2, 0, ain, 0};

    cumo_na_ndloop_store_rarray(&ndf, self, rary);
    return self;
}
