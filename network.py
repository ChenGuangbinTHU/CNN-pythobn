import matplotlib.pyplot as plt

class Network(object):
    def __init__(self):
        self.layer_list = []
        self.params = []
        self.num_layers = 0

    def add(self, layer):
        self.layer_list.append(layer)
        self.num_layers += 1

    def forward(self, input,show):
        output = input
        for i in range(self.num_layers):
            # print(output.shape)
            
            if(i == 1 and show):
                plt.axis('off')
                print(output)
                fig3,ax3 = plt.subplots(nrows=1,ncols=4)
                cnt = 0
                for k in output[0:4]:
                    ax3[cnt].imshow(k[0],cmap='gray_r')
                    cnt = cnt+1
                
                plt.set_cmap('gray')
                plt.show()
                # exit(0)
            output = self.layer_list[i].forward(output)
        

        return output

    def backward(self, grad_output):
        grad_input = grad_output
        for i in range(self.num_layers - 1, -1, -1):
            grad_input = self.layer_list[i].backward(grad_input)

    def update(self, config):
        for i in range(self.num_layers):
            if self.layer_list[i].trainable:
                self.layer_list[i].update(config)
