class Input:
    '''
    Kelas yang mengimplementasikan layer Input 
    pada ANN.
    '''  
    def __init__(self, shape=None):
      '''
      Konstruktor kelas Input.
      
      Parameters:
          shape: (int, int, int)
              Ukuran input yang akan dimasukan, dalam format (channel_input, lebar, tinggi)
      '''
      self.layerType  = "Input"
      self.shape = (None,) + shape
      