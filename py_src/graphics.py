import matplotlib.pyplot as plt
import pandas as pd
import traceback







class Visualizer:
    def __init__(self, sensor_data: pd.DataFrame, metadata: dict, label_data: pd.DataFrame) -> None:
        self.sensor_data = sensor_data
        self.sub = metadata['sub']
        self.tsk = metadata['tsk']
        self.run = metadata['run']
        self.label_data = label_data
        self.presets = { # 0: Plot title            1: x label          2: legends list
            'accel': ['Acceleration over time', 'acceleration ($g$)', ['AccX', 'AccY', 'AccZ']],
            'gyro': ['Angular velocity over time', r'angular velocity ($\frac{^o}{s}$)', ['GyrX', 'GyrY', 'GyrZ']], 
            'euler': ['Euler angles over time', 'angle ($^o$)', ['EulerX', 'EulerY' , 'EulerZ']]
        }
        self.fig, self.axs = plt.subplots(len(self.presets), sharex=True)
        for _ in self.axs:
            _.label_outer()
        pass
    def generate_graphs(self):
        self.fig.suptitle(f'Subject: {self.sub}, Task: {self.tsk}, Run: {self.run}')
        # self.axs[-1].set_xlabel('time ($s$)') 
        for index, value in enumerate(self.presets.values()):
            ax = self.axs[index]
            # ax.set_title(value[0])
            ax.set_ylabel(value[1])
            for i in value[2]:
                line, = ax.plot(self.sensor_data['TimeStamp(s)'], self.sensor_data[i])
                line.set_label(i)
                ax.legend(loc=1)
            query_exp = f'`Task Code (Task ID)` == @self.tsk and `Trial ID` == @self.run'
            row = self.label_data.query(query_exp)
            if not row.empty: 
                self._add_marks(ax, row)
        plt.show(block=False)
        return

    def _add_marks(self, ax, row):
        time = self.sensor_data['TimeStamp(s)']
        onset_frame = row.iat[0, 3]
        impact_frame = row.iat[0, 4]
        ax.axvline(x=time[onset_frame], color = 'r')
        ax.axvline(x=time[impact_frame], color = 'r')

        




if __name__ == '__main__':

    from preprocess import Loader
    print('*-'*20 + 'Visualización de datos' + '-*'*20)
    while True:
        
        # sub = int(input('ID del sujeto: '))
        # tsk = int(input('ID de la actividad: '))
        # run = int(input('ID del intento: '))

        sub, run, tsk = 7, 1, 1
        
        
        loader = Loader()


        sensor_data, sensor_metadata = loader.load_sensor_data(sub, tsk, run)
        label_data = loader.load_label_data(sub)


        visualizer = Visualizer(sensor_data, sensor_metadata, label_data)
        visualizer.generate_graphs()

        input()






