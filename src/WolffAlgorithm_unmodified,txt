    def WolffAlgorithm(self,
                       times,
                       beta,
                       initial_energy: Optional[float64] = None,
                       spinenergy_call: Optional[bool] = False,
                       relax_call: Optional[bool] = False) -> ndarray | None:
        """
            TODO : write docstring
        """
        try:
            if self.SumThreadsAlive is False and relax_call is False:
                self.__LaunchSumThreads__()
            if initial_energy is not None:
                cur_itt_energy = initial_energy
            else:
                cur_itt_energy = np.float64(0)
            netSE_mtx = self.ZERO_mtx
            node_q = LLQueue[Node]()
            for itt_num in range(times):
                # pick random point
                while True:
                    # pick random point on array and flip spin
                    # randint is very slow so dont use it when you need to
                    # call it successively, ever please...
                    rand_xy = np.array([math.trunc((random.random()) *
                                        (self.Lshape[0] - 1)),
                                        math.trunc((random.random()) *
                                        (self.Lshape[1] - 1))])
                    select_node = self[rand_xy]
                    if select_node.get_spin() == 0:
                        continue
                    else:
                        node_q.push(select_node)
                        break
                try:
                    while True:
                        cur_node = node_q.pop()
                        # check neighbors
                        for nbr in cur_node:
                            if nbr.get_spin == 0 or node_q.IsInQueue(nbr) is True:
                                continue
                            nbrs_Energy = 0
                            for Enbr in nbr:
                                nbrs_Energy += Enbr.get_spin()
                            dE = -2*nbr.flip_test()*nbrs_Energy*beta*self.J
                            balcond = 1 - np.exp(dE, dtype=np.float64)

                            if (nbr.get_spin() == -cur_node.get_spin()
                                    and random.random() < balcond):
                                cur_itt_energy += dE
                                nbr.flip_spin()
                                node_q.push(nbr)
                except QueueEmpty:
                    # exit while loop when queue is empty
                    pass
                if relax_call is False:
                    cur_itt_sum = self.GetMagnitization()
                    netSE_mtx[itt_num, 0] = cur_itt_sum
                    netSE_mtx[itt_num, 1] = cur_itt_energy
            # for itt_num in range(times): end
            if spinenergy_call is False:
                self.__StopSumThreads__()
            if relax_call is False:
                return(netSE_mtx)
        except Exception:
            PE.PrintException()