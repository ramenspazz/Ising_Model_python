def WolffAlgorithm(self,
                    times,
                    beta,
                    Ms: ndarray,
                    Es: ndarray,
                    initial_energy: Optional[float64] = None,
                    spinenergy_call: Optional[bool] = False,
                    relax_call: Optional[bool] = False,
                    plot: Optional[bool] = False) -> ndarray | None:
    """
        TODO : write docstring
    """
    try:
        if self.SumThreadsAlive is False and relax_call is False:
            if np.array_equiv(self.ZERO_mtx.shape,
                                np.array([times, 2])) is False:
                self.set_time(times)
            self.__LaunchSumThreads__()
            self.__LaunchPathThreads__()
        if initial_energy is not None:
            cur_itt_energy = initial_energy
        else:
            cur_itt_energy = np.float64(0)
        if spinenergy_call is False:
            inF.print_stdout(
                "WolffAlgorithm is 000.0% complete...")
        netSE_mtx = self.ZERO_mtx
        balcond = 1-np.exp(-2*beta*self.J, dtype=np.float64)

        for itt_num in range(times):
            if spinenergy_call is False:
                inF.print_stdout(
                    f"WolffAlgorithm is {100 * itt_num/times:.2f}%"
                    f" complete...")
            # pick random point
            while True:
                # pick random point on array
                # randint is very slow so dont use it when you need to
                # call it successively, ever please...
                rand_xy = np.array([math.trunc((random.random()) *
                                    (self.Lshape[0] - 1)),
                                    math.trunc((random.random()) *
                                    (self.Lshape[1] - 1))])
                rand_node = self[rand_xy]
                if rand_node.get_spin() == 0:
                    continue
                else:
                    self.cluster.put(rand_node.get_index())
                    break
            # push the random node to the work queue
            for nbr in rand_node:
                self.work_queue_path.put(nbr.get_index())
            self.was_seen[rand_node] = rand_node
            # wait for cluster to be generated
            self.GetCluster(balcond)
            self.was_seen.clear()
            try:
                # evaluate energy change path integral (discrete sum? lol)
                cur = self.LinkedLat[self.cluster.get(block=False)]
                while True:
                    nbr_sum = float64(0)
                    for nbr in cur:
                        nbr_sum += nbr.get_spin()
                    cur_itt_energy += nbr_sum*cur.flip_spin()*self.J
                    cur = self.LinkedLat[self.cluster.get(block=False)]
            except queue.Empty:
                # exit while loop when queue is empty
                pass
            if relax_call is False:
                # The purpose of this check is to get reasonable results
                # after a few itterations
                cur_itt_sum = self.GetMagnitization()
                netSE_mtx[itt_num, 0] = cur_itt_sum
                netSE_mtx[itt_num, 1] = cur_itt_energy
                if itt_num > int(times*0.6):
                    Ms[0] += cur_itt_sum
                    Ms[1] += cur_itt_sum**2
                    Es[0] += cur_itt_energy
                    Es[1] += cur_itt_energy**2
        # for itt_num in range(times): end

        if spinenergy_call is False:
            inF.print_stdout(
                "WolffAlgorithm is 100% complete...")
        if spinenergy_call is False:
            self.__StopSumThreads__()
            self.__StopPathThreads__()
        if plot is True:
            self.PlotItterations(netSE_mtx, beta, save=False,
                                    auto_plot=True)
        if relax_call is False:
            return([Ms, Es, netSE_mtx])
    except Exception:
        PE.PrintException()