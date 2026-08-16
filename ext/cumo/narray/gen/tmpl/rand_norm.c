typedef struct {
    dtype mu;
    rtype sigma;
    u_int64_t seed;
    u_int64_t offset;
} randn_opt_t;

void <%="cumo_#{c_iter}_index_kernel_launch"%>(char *p1, size_t *idx1, uint64_t seed, uint64_t offset, dtype mu, rtype sigma, uint64_t n);
void <%="cumo_#{c_iter}_stride_kernel_launch"%>(char *p1, ssize_t s1, uint64_t seed, uint64_t offset, dtype mu, rtype sigma, uint64_t n);

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t   i;
    char    *p1;
    ssize_t  s1;
    size_t  *idx1;
    dtype    mu;
    rtype    sigma;
    randn_opt_t *g;

    CUMO_INIT_COUNTER(lp, i);
    CUMO_INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    g = (randn_opt_t*)(lp->opt_ptr);
    mu = g->mu;
    sigma = g->sigma;

    {
        size_t n = i;
        if (idx1) {
            <%="cumo_#{c_iter}_index_kernel_launch"%>(p1,idx1,g->seed,g->offset,mu,sigma,n);
        } else {
            <%="cumo_#{c_iter}_stride_kernel_launch"%>(p1,s1,g->seed,g->offset,mu,sigma,n);
        }
        g->offset += n;
    }
}

/*
  Generates random numbers from the normal distribution on self narray
  using Box-Muller Transformation.
  @overload rand_norm([mu,[sigma]])
  @param [Numeric] mu  mean of normal distribution. (default=0)
  @param [Numeric] sigma  standard deviation of normal distribution. (default=1)
  @return [Cumo::<%=class_name%>] self.
  @example
    Cumo::DFloat.new(5,5).rand_norm
    # => Cumo::DFloat#shape=[5,5]
    #    [[-0.310396, -0.0343281, 0.175595, -2.28038, 0.50386],
    #     [0.559625, -0.0749621, 0.969067, -0.23566, -0.458208],
    #     [0.566074, 1.2851, -1.86667, -0.0312267, 1.24331],
    #     [1.36889, -1.07531, -0.0157597, -1.44813, -1.30887],
    #     [0.697989, -0.330042, -0.770826, -0.494565, 3.17022]]

    Cumo::DFloat.new(5,5).rand_norm(10,0.1)
    # => Cumo::DFloat#shape=[5,5]
    #    [[9.96896, 9.99657, 10.0176, 9.77196, 10.0504],
    #     [10.056, 9.9925, 10.0969, 9.97643, 9.95418],
    #     [10.0566, 10.1285, 9.81333, 9.99688, 10.1243],
    #     [10.1369, 9.89247, 9.99842, 9.85519, 9.86911],
    #     [10.0698, 9.967, 9.92292, 9.95054, 10.317]]

    Cumo::DComplex.new(3,3).rand_norm(5+5i)
    # => Cumo::DComplex#shape=[3,3]
    #    [[4.6896+4.60233i, 4.96567+4.64886i, 5.17559+4.19631i],
    #     [2.71962+4.57164i, 5.50386+5.9536i, 5.55962+5.70959i],
    #     [4.92504+6.0447i, 5.96907+5.35384i, 4.76434+5.41498i]]
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE *args, VALUE self)
{
    int n;
    randn_opt_t g;
    VALUE v1=Qnil, v2=Qnil;
    cumo_ndfunc_arg_in_t ain[1] = {{CUMO_OVERWRITE,0}};
    cumo_ndfunc_t ndf = {<%=c_iter%>, CUMO_FULL_LOOP, 1,0, ain,0};

    n = rb_scan_args(argc, args, "02", &v1, &v2);
    if (n == 0) {
        g.mu = m_zero;
    } else {
        g.mu = m_num_to_data(v1);
    }
    if (n == 2) {
        g.sigma = NUM2DBL(v2);
    } else {
        g.sigma = 1;
    }
    g.seed = cumo_cuda_rand_seed();
    g.offset = cumo_cuda_rand_offset();
    cumo_na_ndloop3(&ndf, &g, 1, self);
    cumo_cuda_rand_set_offset(g.offset);
    return self;
}
