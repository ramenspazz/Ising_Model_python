# def set_basis(self, input_basis: NDArray[Float]) -> None:
#     """
#         Parameters
#         ----------
#         input_basis : `NDArray[Float]`
#             Specifies a basis for the crystal.
#     """
#     for basis_vector in input_basis:
#         self.basis_arr.append(basis_vector)

# def get_root(self) -> Node:
#     """Returns the origin (A.K.A. root) node."""
#     return(self.origin_node)

# def MetropolisSpinEnergy(self,
#                          Beta: list | ndarray,
#                          times: int,
#                          save: Optional[bool] = False,
#                          auto_plot: Optional[bool] = True) -> ndarray:
#     """
#         Purpose
#         -------
#         Computes the energy of the lattice at thermal equlibrium and
#         relates this to T=J/(Beta*J*k) = 1/(Beta*k) ~ 1/(Beta*J)
#     """
#     try:
#         inF.print_stdout(
#             "get_spin_energy is 000.0% complete...\r")
#         if np.array_equiv(self.ZERO_mtx.shape,
#                           np.array([times, 2])) is False:
#             self.set_time(times)
#         netSE_mtx = self.ZERO_mtx
#         magnitization = np.zeros(len(Beta), float64)
#         E_mean = np.zeros(len(Beta), float64)
#         E_std = np.zeros(len(Beta), float64)

#         # multithreading object declarations
#         qsize = self.LinkedLat.tc
#         result_queue = MLTPqueue.ThQueue(qsize)
#         ready_sum = WaitListLock(qsize)
#         finished_sum = mltp.Event()

#         thread_pool_sum: list[mltp.Process] = []
#         for th_num in range(self.LinkedLat.tc):
#             thread_pool_sum.append(mltp.Process(
#                 target=self.LinkedLat.Sum_Worker,
#                 args=(self.LinkedLat.bounds[th_num],
#                       th_num,
#                       result_queue,
#                       ready_sum,
#                       finished_sum)))
#             thread_pool_sum[th_num].start()

#         # flip_num = math.ceil(np.log2(len(self.internal_arr)))

#         start_time = time.time()
#         energy = self.LinkedLat.__threadlauncher__(
#                 self.LinkedLat.Energy_Worker, True)

#         for i, beta in enumerate(Beta):
#             inF.print_stdout(
#                 f"get_spin_energy is {100 * i / len(Beta) :.1f}%"
#                 f" complete...")
#             netSE_mtx = self.ZERO_mtx
#             for itt_num in range(times):
#                 psum = 0
#                 # select spins to flip
#                 while True:
#                     # pick random point on array and flip spin
#                     # randint is very slow so dont use it, ever please...
#                     rand_xy = np.array([math.trunc((random.random()) *
#                                        (self.Lshape[0] - 1)),
#                                        math.trunc((random.random()) *
#                                                   (self.Lshape[1] - 1))])
#                     node_i = self[rand_xy]
#                     if node_i.get_spin() == 0:
#                         continue
#                     else:
#                         break
#                 # compute change in energy
#                 nbrs_Energy: np.int64 = 0
#                 for neighbor in node_i:
#                     if neighbor.get_spin() == 0:
#                         continue
#                     nbrs_Energy += neighbor.get_spin()
#                 dE = 2 * self.J * node_i.flip_test() * nbrs_Energy
#                 if dE < 0:
#                     node_i.flip_spin()
#                 elif (random.random() > np.exp(-dE)):
#                     node_i.flip_spin()

#                 # begin calculating total spin of lattice
#                 ready_sum.Check()
#                 ready_sum.Start_Threads()
#                 for j in range(qsize):
#                     psum += result_queue.get(block=True)

#                 energy += dE
#                 netSE_mtx[itt_num, 0] = psum
#                 netSE_mtx[itt_num, 1] = energy
#             # for itt_num in range(times): end
#             magnitization[i] = DA.data_mean(netSE_mtx[-times:, 0])
#             E_mean[i] = DA.data_mean(netSE_mtx[-times:, 1])
#             E_std[i] = DA.std_dev(netSE_mtx[-times:, 1], E_mean[i])
#         # for i, beta in enumerate(Beta): end

#         finished_sum.set()
#         ready_sum.Start_Threads()
#         for t in thread_pool_sum:
#             t.join()

#         end_time = time.time()
#         inF.print_stdout(
#             f'get_spin_energy is 100% complete in '
#             f'{end_time-start_time:.8f} seconds!',
#             end='\n')
#         self.PlotSpinEnergy(Beta,
#                               magnitization,
#                               E_std,
#                               save=save,
#                               auto_plot=auto_plot,
#                               times=times)
#         return(netSE_mtx)
#     except KeyboardInterrupt:
#         print('\nKeyboard Inturrupt, exiting...\n')
#         finished_sum.set()
#         ready_sum.Start_Threads()
#         for t in thread_pool_sum:
#             t.terminate()
#         exit()
#     except Exception:
#         PE.PrintException()



# # multithreading object declarations for sum
# result_queue_SE = MLTPqueue.ThQueue(qsize)
# ready_SE = WaitListLock(qsize)
# finished_SE = mltp.Event()

# thread_pool_SE: list[mltp.Process] = []
# for th_num in range(self.LinkedLat.tc):
#     thread_pool_SE.append(mltp.Process(
#         target=self.LinkedLat.SpinEnergy_Worker,
#         args=(self.LinkedLat.bounds[th_num],
#               th_num,
#               result_queue_SE,
#               ready_SE,
#               finished_SE)))
#     thread_pool_SE[th_num].start()
