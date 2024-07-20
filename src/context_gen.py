import tenseal as ts


class TenSEALContext:
    @staticmethod
    def create_context():
        context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, -1, [60, 40, 40, 60])
        context.global_scale = 2**40
        context.generate_galois_keys()
        return context

    def create_machine_learning_context():
        context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, -1,
                             [40, 21, 21, 21, 21, 21, 21, 40])
        context.global_scale = 2**21
        context.generate_galois_keys()
        return context
