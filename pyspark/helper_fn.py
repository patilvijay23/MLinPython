def group_by_and_rename(df,dims,metrics):
    '''
        # Method is intended to groupby and rename with <aggregation>_<col name>
        # Note: Only one aggreagtion can be applied to one column, in case more than one are
        #       passed, the last one will get applied
        :param df - spark dataframe
        :param dims - list/tuple of the metrics to group by, e.g. ['col1','col2'] or ('col1','col2')
        :param metrics - dictionnary: {a: b} where a = method ('sum','max',..), and
                         b = list of metric(s) ['col3'] or ['col3','col4']
        :return - df
    '''
    # preparing the agg dic
    d = {}
    for s in metrics:
        for m in metrics[s]:
            d[m] = s
    # checking exception of one dimension only
    if type(dims)==str:
        df = df.groupby(dims).agg(d)
    elif type(dims) in [list,tuple]:
        if len(dims)==1:
            df = df.groupby(dims[0]).agg(d)
        else:
            dims = tuple(dims)
            df = df.groupby(*dims).agg(d)
    # renaming the columns
    for s in metrics:
        for m in metrics[s]:
            df = df.withColumnRenamed( s+'('+m+')',s+'_'+m)
    return df