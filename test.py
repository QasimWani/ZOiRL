x = []
for bid in range(9):
    x.append(self.init_updates['c_Csto_init'][bid]-env.buildings[f'Building_{bid+1}'].cooling_storage_soc[-1]/env.buildings[f'Building_{bid+1}'].cooling_storage.capacity)
y= []
for bid in range(9):
    y.append(
        self.init_updates['c_Hsto_init'][bid] - env.buildings[f'Building_{bid + 1}'].dhw_storage_soc[-1] /
        env.buildings[f'Building_{bid + 1}'].dhw_storage.capacity)
z = []
for bid in range(9):
    z.append(
        self.init_updates['c_bat_init'][bid] - env.buildings[f'Building_{bid + 1}'].electrical_storage_soc[-1] /
        env.buildings[f'Building_{bid + 1}'].electrical_storage.capacity)