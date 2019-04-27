from demo_loader import *

brain_params, demo_buffer = demo_to_buffer('Gameplay_1.demo',1)

print(brain_params)
print(          demo_buffer.update_buffer)
# for i in demo_buffer[0]['vector_obs']:
#     print(i)