'''
Converted from C++ to Python by John Kim and ChatGPT4o
Original C++ code from https://github.com/madcowswe/ESKF/blob/master/src/main.cpp
'''

def unrolledFPFt(Pin, Pnew, dt, dVel_dTheta, dVel_dAccelBias, dTheta_dTheta):
    x0 = Pin[1,0] + Pin[4,0]*dt + dt*(Pin[3,1] + Pin[4,3]*dt)
    x1 = Pin[2,0] + Pin[5,0]*dt + dt*(Pin[3,2] + Pin[5,3]*dt)
    x2 = Pin[10,3]*dVel_dAccelBias[0,1] + Pin[11,3]*dVel_dAccelBias[0,2] + Pin[3,3] + Pin[6,3]*dVel_dTheta[0,0] + Pin[7,3]*dVel_dTheta[0,1] + Pin[8,3]*dVel_dTheta[0,2] + Pin[9,3]*dVel_dAccelBias[0,0]
    x3 = Pin[10,0]*dVel_dAccelBias[0,1] + Pin[11,0]*dVel_dAccelBias[0,2] + Pin[3,0] + Pin[6,0]*dVel_dTheta[0,0] + Pin[7,0]*dVel_dTheta[0,1] + Pin[8,0]*dVel_dTheta[0,2] + Pin[9,0]*dVel_dAccelBias[0,0] + dt*x2
    x4 = Pin[10,3]*dVel_dAccelBias[1,1] + Pin[11,3]*dVel_dAccelBias[1,2] + Pin[4,3] + Pin[6,3]*dVel_dTheta[1,0] + Pin[7,3]*dVel_dTheta[1,1] + Pin[8,3]*dVel_dTheta[1,2] + Pin[9,3]*dVel_dAccelBias[1,0]
    x5 = Pin[10,0]*dVel_dAccelBias[1,1] + Pin[11,0]*dVel_dAccelBias[1,2] + Pin[4,0] + Pin[6,0]*dVel_dTheta[1,0] + Pin[7,0]*dVel_dTheta[1,1] + Pin[8,0]*dVel_dTheta[1,2] + Pin[9,0]*dVel_dAccelBias[1,0] + dt*x4
    x6 = Pin[10,3]*dVel_dAccelBias[2,1] + Pin[11,3]*dVel_dAccelBias[2,2] + Pin[5,3] + Pin[6,3]*dVel_dTheta[2,0] + Pin[7,3]*dVel_dTheta[2,1] + Pin[8,3]*dVel_dTheta[2,2] + Pin[9,3]*dVel_dAccelBias[2,0]
    x7 = Pin[10,0]*dVel_dAccelBias[2,1] + Pin[11,0]*dVel_dAccelBias[2,2] + Pin[5,0] + Pin[6,0]*dVel_dTheta[2,0] + Pin[7,0]*dVel_dTheta[2,1] + Pin[8,0]*dVel_dTheta[2,2] + Pin[9,0]*dVel_dAccelBias[2,0] + dt*x6
    x8 = Pin[12,3]*dt
    x9 = Pin[6,3]*dTheta_dTheta[0,0] + Pin[7,3]*dTheta_dTheta[0,1] + Pin[8,3]*dTheta_dTheta[0,2] - x8
    x10 = -Pin[12,0]*dt + Pin[6,0]*dTheta_dTheta[0,0] + Pin[7,0]*dTheta_dTheta[0,1] + Pin[8,0]*dTheta_dTheta[0,2] + dt*x9
    x11 = Pin[13,3]*dt
    x12 = Pin[6,3]*dTheta_dTheta[1,0] + Pin[7,3]*dTheta_dTheta[1,1] + Pin[8,3]*dTheta_dTheta[1,2] - x11
    x13 = -Pin[13,0]*dt + Pin[6,0]*dTheta_dTheta[1,0] + Pin[7,0]*dTheta_dTheta[1,1] + Pin[8,0]*dTheta_dTheta[1,2] + dt*x12
    x14 = Pin[14,3]*dt
    x15 = Pin[6,3]*dTheta_dTheta[2,0] + Pin[7,3]*dTheta_dTheta[2,1] + Pin[8,3]*dTheta_dTheta[2,2] - x14
    x16 = -Pin[14,0]*dt + Pin[6,0]*dTheta_dTheta[2,0] + Pin[7,0]*dTheta_dTheta[2,1] + Pin[8,0]*dTheta_dTheta[2,2] + dt*x15
    x17 = Pin[9,0] + Pin[9,3]*dt
    x18 = Pin[10,0] + Pin[10,3]*dt
    x19 = Pin[11,0] + Pin[11,3]*dt
    x20 = Pin[12,0] + x8
    x21 = Pin[13,0] + x11
    x22 = Pin[14,0] + x14
    x23 = Pin[2,1] + Pin[5,1]*dt + dt*(Pin[4,2] + Pin[5,4]*dt)
    x24 = Pin[10,1]*dVel_dAccelBias[0,1] + Pin[11,1]*dVel_dAccelBias[0,2] + Pin[3,1] + Pin[6,1]*dVel_dTheta[0,0] + Pin[7,1]*dVel_dTheta[0,1] + Pin[8,1]*dVel_dTheta[0,2] + Pin[9,1]*dVel_dAccelBias[0,0] + dt*(Pin[10,4]*dVel_dAccelBias[0,1] + Pin[11,4]*dVel_dAccelBias[0,2] + Pin[4,3] + Pin[6,4]*dVel_dTheta[0,0] + Pin[7,4]*dVel_dTheta[0,1] + Pin[8,4]*dVel_dTheta[0,2] + Pin[9,4]*dVel_dAccelBias[0,0])
    x25 = Pin[10,4]*dVel_dAccelBias[1,1] + Pin[11,4]*dVel_dAccelBias[1,2] + Pin[4,4] + Pin[6,4]*dVel_dTheta[1,0] + Pin[7,4]*dVel_dTheta[1,1] + Pin[8,4]*dVel_dTheta[1,2] + Pin[9,4]*dVel_dAccelBias[1,0]
    x26 = Pin[10,1]*dVel_dAccelBias[1,1] + Pin[11,1]*dVel_dAccelBias[1,2] + Pin[4,1] + Pin[6,1]*dVel_dTheta[1,0] + Pin[7,1]*dVel_dTheta[1,1] + Pin[8,1]*dVel_dTheta[1,2] + Pin[9,1]*dVel_dAccelBias[1,0] + dt*x25
    x27 = Pin[10,4]*dVel_dAccelBias[2,1] + Pin[11,4]*dVel_dAccelBias[2,2] + Pin[5,4] + Pin[6,4]*dVel_dTheta[2,0] + Pin[7,4]*dVel_dTheta[2,1] + Pin[8,4]*dVel_dTheta[2,2] + Pin[9,4]*dVel_dAccelBias[2,0]
    x28 = Pin[10,1]*dVel_dAccelBias[2,1] + Pin[11,1]*dVel_dAccelBias[2,2] + Pin[5,1] + Pin[6,1]*dVel_dTheta[2,0] + Pin[7,1]*dVel_dTheta[2,1] + Pin[8,1]*dVel_dTheta[2,2] + Pin[9,1]*dVel_dAccelBias[2,0] + dt*x27
    x29 = Pin[12,4]*dt
    x30 = Pin[6,4]*dTheta_dTheta[0,0] + Pin[7,4]*dTheta_dTheta[0,1] + Pin[8,4]*dTheta_dTheta[0,2] - x29
    x31 = -Pin[12,1]*dt + Pin[6,1]*dTheta_dTheta[0,0] + Pin[7,1]*dTheta_dTheta[0,1] + Pin[8,1]*dTheta_dTheta[0,2] + dt*x30
    x32 = Pin[13,4]*dt
    x33 = Pin[6,4]*dTheta_dTheta[1,0] + Pin[7,4]*dTheta_dTheta[1,1] + Pin[8,4]*dTheta_dTheta[1,2] - x32
    x34 = -Pin[13,1]*dt + Pin[6,1]*dTheta_dTheta[1,0] + Pin[7,1]*dTheta_dTheta[1,1] + Pin[8,1]*dTheta_dTheta[1,2] + dt*x33
    x35 = Pin[14,4]*dt
    x36 = Pin[6,4]*dTheta_dTheta[2,0] + Pin[7,4]*dTheta_dTheta[2,1] + Pin[8,4]*dTheta_dTheta[2,2] - x35
    x37 = -Pin[14,1]*dt + Pin[6,1]*dTheta_dTheta[2,0] + Pin[7,1]*dTheta_dTheta[2,1] + Pin[8,1]*dTheta_dTheta[2,2] + dt*x36
    x38 = Pin[9,1] + Pin[9,4]*dt
    x39 = Pin[10,1] + Pin[10,4]*dt
    x40 = Pin[11,1] + Pin[11,4]*dt
    x41 = Pin[12,1] + x29
    x42 = Pin[13,1] + x32
    x43 = Pin[14,1] + x35
    x44 = Pin[10,2]*dVel_dAccelBias[0,1] + Pin[11,2]*dVel_dAccelBias[0,2] + Pin[3,2] + Pin[6,2]*dVel_dTheta[0,0] + Pin[7,2]*dVel_dTheta[0,1] + Pin[8,2]*dVel_dTheta[0,2] + Pin[9,2]*dVel_dAccelBias[0,0] + dt*(Pin[10,5]*dVel_dAccelBias[0,1] + Pin[11,5]*dVel_dAccelBias[0,2] + Pin[5,3] + Pin[6,5]*dVel_dTheta[0,0] + Pin[7,5]*dVel_dTheta[0,1] + Pin[8,5]*dVel_dTheta[0,2] + Pin[9,5]*dVel_dAccelBias[0,0])
    x45 = Pin[10,2]*dVel_dAccelBias[1,1] + Pin[11,2]*dVel_dAccelBias[1,2] + Pin[4,2] + Pin[6,2]*dVel_dTheta[1,0] + Pin[7,2]*dVel_dTheta[1,1] + Pin[8,2]*dVel_dTheta[1,2] + Pin[9,2]*dVel_dAccelBias[1,0] + dt*(Pin[10,5]*dVel_dAccelBias[1,1] + Pin[11,5]*dVel_dAccelBias[1,2] + Pin[5,4] + Pin[6,5]*dVel_dTheta[1,0] + Pin[7,5]*dVel_dTheta[1,1] + Pin[8,5]*dVel_dTheta[1,2] + Pin[9,5]*dVel_dAccelBias[1,0])
    x46 = Pin[10,5]*dVel_dAccelBias[2,1] + Pin[11,5]*dVel_dAccelBias[2,2] + Pin[5,5] + Pin[6,5]*dVel_dTheta[2,0] + Pin[7,5]*dVel_dTheta[2,1] + Pin[8,5]*dVel_dTheta[2,2] + Pin[9,5]*dVel_dAccelBias[2,0]
    x47 = Pin[10,2]*dVel_dAccelBias[2,1] + Pin[11,2]*dVel_dAccelBias[2,2] + Pin[5,2] + Pin[6,2]*dVel_dTheta[2,0] + Pin[7,2]*dVel_dTheta[2,1] + Pin[8,2]*dVel_dTheta[2,2] + Pin[9,2]*dVel_dAccelBias[2,0] + dt*x46
    x48 = Pin[12,5]*dt
    x49 = Pin[6,5]*dTheta_dTheta[0,0] + Pin[7,5]*dTheta_dTheta[0,1] + Pin[8,5]*dTheta_dTheta[0,2] - x48
    x50 = -Pin[12,2]*dt + Pin[6,2]*dTheta_dTheta[0,0] + Pin[7,2]*dTheta_dTheta[0,1] + Pin[8,2]*dTheta_dTheta[0,2] + dt*x49
    x51 = Pin[13,5]*dt
    x52 = Pin[6,5]*dTheta_dTheta[1,0] + Pin[7,5]*dTheta_dTheta[1,1] + Pin[8,5]*dTheta_dTheta[1,2] - x51
    x53 = -Pin[13,2]*dt + Pin[6,2]*dTheta_dTheta[1,0] + Pin[7,2]*dTheta_dTheta[1,1] + Pin[8,2]*dTheta_dTheta[1,2] + dt*x52
    x54 = Pin[14,5]*dt
    x55 = Pin[6,5]*dTheta_dTheta[2,0] + Pin[7,5]*dTheta_dTheta[2,1] + Pin[8,5]*dTheta_dTheta[2,2] - x54
    x56 = -Pin[14,2]*dt + Pin[6,2]*dTheta_dTheta[2,0] + Pin[7,2]*dTheta_dTheta[2,1] + Pin[8,2]*dTheta_dTheta[2,2] + dt*x55
    x57 = Pin[9,2] + Pin[9,5]*dt
    x58 = Pin[10,2] + Pin[10,5]*dt
    x59 = Pin[11,2] + Pin[11,5]*dt
    x60 = Pin[12,2] + x48
    x61 = Pin[13,2] + x51
    x62 = Pin[14,2] + x54
    x63 = Pin[10,9]*dVel_dAccelBias[0,1] + Pin[11,9]*dVel_dAccelBias[0,2] + Pin[9,3] + Pin[9,6]*dVel_dTheta[0,0] + Pin[9,7]*dVel_dTheta[0,1] + Pin[9,8]*dVel_dTheta[0,2] + Pin[9,9]*dVel_dAccelBias[0,0]
    x64 = Pin[10,10]*dVel_dAccelBias[0,1] + Pin[10,3] + Pin[10,6]*dVel_dTheta[0,0] + Pin[10,7]*dVel_dTheta[0,1] + Pin[10,8]*dVel_dTheta[0,2] + Pin[10,9]*dVel_dAccelBias[0,0] + Pin[11,10]*dVel_dAccelBias[0,2]
    x65 = Pin[11,10]*dVel_dAccelBias[0,1] + Pin[11,11]*dVel_dAccelBias[0,2] + Pin[11,3] + Pin[11,6]*dVel_dTheta[0,0] + Pin[11,7]*dVel_dTheta[0,1] + Pin[11,8]*dVel_dTheta[0,2] + Pin[11,9]*dVel_dAccelBias[0,0]
    x66 = Pin[10,9]*dVel_dAccelBias[1,1] + Pin[11,9]*dVel_dAccelBias[1,2] + Pin[9,4] + Pin[9,6]*dVel_dTheta[1,0] + Pin[9,7]*dVel_dTheta[1,1] + Pin[9,8]*dVel_dTheta[1,2] + Pin[9,9]*dVel_dAccelBias[1,0]
    x67 = Pin[10,10]*dVel_dAccelBias[1,1] + Pin[10,4] + Pin[10,6]*dVel_dTheta[1,0] + Pin[10,7]*dVel_dTheta[1,1] + Pin[10,8]*dVel_dTheta[1,2] + Pin[10,9]*dVel_dAccelBias[1,0] + Pin[11,10]*dVel_dAccelBias[1,2]
    x68 = Pin[11,10]*dVel_dAccelBias[1,1] + Pin[11,11]*dVel_dAccelBias[1,2] + Pin[11,4] + Pin[11,6]*dVel_dTheta[1,0] + Pin[11,7]*dVel_dTheta[1,1] + Pin[11,8]*dVel_dTheta[1,2] + Pin[11,9]*dVel_dAccelBias[1,0]
    x69 = Pin[10,6]*dVel_dAccelBias[1,1] + Pin[11,6]*dVel_dAccelBias[1,2] + Pin[6,4] + Pin[6,6]*dVel_dTheta[1,0] + Pin[7,6]*dVel_dTheta[1,1] + Pin[8,6]*dVel_dTheta[1,2] + Pin[9,6]*dVel_dAccelBias[1,0]
    x70 = Pin[10,7]*dVel_dAccelBias[1,1] + Pin[11,7]*dVel_dAccelBias[1,2] + Pin[7,4] + Pin[7,6]*dVel_dTheta[1,0] + Pin[7,7]*dVel_dTheta[1,1] + Pin[8,7]*dVel_dTheta[1,2] + Pin[9,7]*dVel_dAccelBias[1,0]
    x71 = Pin[10,8]*dVel_dAccelBias[1,1] + Pin[11,8]*dVel_dAccelBias[1,2] + Pin[8,4] + Pin[8,6]*dVel_dTheta[1,0] + Pin[8,7]*dVel_dTheta[1,1] + Pin[8,8]*dVel_dTheta[1,2] + Pin[9,8]*dVel_dAccelBias[1,0]
    x72 = dVel_dAccelBias[0,0]*x66 + dVel_dAccelBias[0,1]*x67 + dVel_dAccelBias[0,2]*x68 + dVel_dTheta[0,0]*x69 + dVel_dTheta[0,1]*x70 + dVel_dTheta[0,2]*x71 + x4
    x73 = Pin[10,9]*dVel_dAccelBias[2,1] + Pin[11,9]*dVel_dAccelBias[2,2] + Pin[9,5] + Pin[9,6]*dVel_dTheta[2,0] + Pin[9,7]*dVel_dTheta[2,1] + Pin[9,8]*dVel_dTheta[2,2] + Pin[9,9]*dVel_dAccelBias[2,0]
    x74 = Pin[10,10]*dVel_dAccelBias[2,1] + Pin[10,5] + Pin[10,6]*dVel_dTheta[2,0] + Pin[10,7]*dVel_dTheta[2,1] + Pin[10,8]*dVel_dTheta[2,2] + Pin[10,9]*dVel_dAccelBias[2,0] + Pin[11,10]*dVel_dAccelBias[2,2]
    x75 = Pin[11,10]*dVel_dAccelBias[2,1] + Pin[11,11]*dVel_dAccelBias[2,2] + Pin[11,5] + Pin[11,6]*dVel_dTheta[2,0] + Pin[11,7]*dVel_dTheta[2,1] + Pin[11,8]*dVel_dTheta[2,2] + Pin[11,9]*dVel_dAccelBias[2,0]
    x76 = Pin[10,6]*dVel_dAccelBias[2,1] + Pin[11,6]*dVel_dAccelBias[2,2] + Pin[6,5] + Pin[6,6]*dVel_dTheta[2,0] + Pin[7,6]*dVel_dTheta[2,1] + Pin[8,6]*dVel_dTheta[2,2] + Pin[9,6]*dVel_dAccelBias[2,0]
    x77 = Pin[10,7]*dVel_dAccelBias[2,1] + Pin[11,7]*dVel_dAccelBias[2,2] + Pin[7,5] + Pin[7,6]*dVel_dTheta[2,0] + Pin[7,7]*dVel_dTheta[1,1] + Pin[8,7]*dVel_dTheta[2,2] + Pin[9,7]*dVel_dAccelBias[2,0]
    x78 = Pin[10,8]*dVel_dAccelBias[2,1] + Pin[11,8]*dVel_dAccelBias[2,2] + Pin[8,5] + Pin[8,6]*dVel_dTheta[2,0] + Pin[8,7]*dVel_dTheta[2,1] + Pin[8,8]*dVel_dTheta[2,2] + Pin[9,8]*dVel_dAccelBias[2,0]
    x79 = dVel_dAccelBias[0,0]*x73 + dVel_dAccelBias[0,1]*x74 + dVel_dAccelBias[0,2]*x75 + dVel_dTheta[0,0]*x76 + dVel_dTheta[0,1]*x77 + dVel_dTheta[0,2]*x78 + x6
    x80 = -Pin[12,9]*dt + Pin[9,6]*dTheta_dTheta[0,0] + Pin[9,7]*dTheta_dTheta[0,1] + Pin[9,8]*dTheta_dTheta[0,2]
    x81 = Pin[10,6]*dTheta_dTheta[0,0] + Pin[10,7]*dTheta_dTheta[0,1] + Pin[10,8]*dTheta_dTheta[0,2] - Pin[12,10]*dt
    x82 = Pin[11,6]*dTheta_dTheta[0,0] + Pin[11,7]*dTheta_dTheta[0,1] + Pin[11,8]*dTheta_dTheta[0,2] - Pin[12,11]*dt
    x83 = -Pin[12,6]*dt + Pin[6,6]*dTheta_dTheta[0,0] + Pin[7,6]*dTheta_dTheta[0,1] + Pin[8,6]*dTheta_dTheta[0,2]
    x84 = -Pin[12,7]*dt + Pin[7,6]*dTheta_dTheta[0,0] + Pin[7,7]*dTheta_dTheta[0,1] + Pin[8,7]*dTheta_dTheta[0,2]
    x85 = -Pin[12,8]*dt + Pin[8,6]*dTheta_dTheta[0,0] + Pin[8,7]*dTheta_dTheta[0,1] + Pin[8,8]*dTheta_dTheta[0,2]
    x86 = dVel_dAccelBias[0,0]*x80 + dVel_dAccelBias[0,1]*x81 + dVel_dAccelBias[0,2]*x82 + dVel_dTheta[0,0]*x83 + dVel_dTheta[0,1]*x84 + dVel_dTheta[0,2]*x85 + x9
    x87 = -Pin[13,9]*dt + Pin[9,6]*dTheta_dTheta[1,0] + Pin[9,7]*dTheta_dTheta[1,1] + Pin[9,8]*dTheta_dTheta[1,2]
    x88 = Pin[10,6]*dTheta_dTheta[1,0] + Pin[10,7]*dTheta_dTheta[1,1] + Pin[10,8]*dTheta_dTheta[1,2] - Pin[13,10]*dt
    x89 = Pin[11,6]*dTheta_dTheta[1,0] + Pin[11,7]*dTheta_dTheta[1,1] + Pin[11,8]*dTheta_dTheta[1,2] - Pin[13,11]*dt
    x90 = -Pin[13,6]*dt + Pin[6,6]*dTheta_dTheta[1,0] + Pin[7,6]*dTheta_dTheta[1,1] + Pin[8,6]*dTheta_dTheta[1,2]
    x91 = -Pin[13,7]*dt + Pin[7,6]*dTheta_dTheta[1,0] + Pin[7,7]*dTheta_dTheta[1,1] + Pin[8,7]*dTheta_dTheta[1,2]
    x92 = -Pin[13,8]*dt + Pin[8,6]*dTheta_dTheta[1,0] + Pin[8,7]*dTheta_dTheta[1,1] + Pin[8,8]*dTheta_dTheta[1,2]
    x93 = dVel_dAccelBias[0,0]*x87 + dVel_dAccelBias[0,1]*x88 + dVel_dAccelBias[0,2]*x89 + dVel_dTheta[0,0]*x90 + dVel_dTheta[0,1]*x91 + dVel_dTheta[0,2]*x92 + x12
    x94 = -Pin[14,9]*dt + Pin[9,6]*dTheta_dTheta[2,0] + Pin[9,7]*dTheta_dTheta[2,1] + Pin[9,8]*dTheta_dTheta[2,2]
    x95 = Pin[10,6]*dTheta_dTheta[2,0] + Pin[10,7]*dTheta_dTheta[2,1] + Pin[10,8]*dTheta_dTheta[2,2] - Pin[14,10]*dt
    x96 = Pin[11,6]*dTheta_dTheta[2,0] + Pin[11,7]*dTheta_dTheta[2,1] + Pin[11,8]*dTheta_dTheta[2,2] - Pin[14,11]*dt
    x97 = -Pin[14,6]*dt + Pin[6,6]*dTheta_dTheta[2,0] + Pin[7,6]*dTheta_dTheta[2,1] + Pin[8,6]*dTheta_dTheta[2,2]
    x98 = -Pin[14,7]*dt + Pin[7,6]*dTheta_dTheta[2,0] + Pin[7,7]*dTheta_dTheta[2,1] + Pin[8,7]*dTheta_dTheta[2,2]
    x99 = -Pin[14,8]*dt + Pin[8,6]*dTheta_dTheta[2,0] + Pin[8,7]*dTheta_dTheta[2,1] + Pin[8,8]*dTheta_dTheta[2,2]
    x100 = dVel_dAccelBias[0,0]*x94 + dVel_dAccelBias[0,1]*x95 + dVel_dAccelBias[0,2]*x96 + dVel_dTheta[0,0]*x97 + dVel_dTheta[0,1]*x98 + dVel_dTheta[0,2]*x99 + x15
    x101 = Pin[12,10]*dVel_dAccelBias[0,1] + Pin[12,11]*dVel_dAccelBias[0,2] + Pin[12,3] + Pin[12,6]*dVel_dTheta[0,0] + Pin[12,7]*dVel_dTheta[0,1] + Pin[12,8]*dVel_dTheta[0,2] + Pin[12,9]*dVel_dAccelBias[0,0]
    x102 = Pin[13,10]*dVel_dAccelBias[0,1] + Pin[13,11]*dVel_dAccelBias[0,2] + Pin[13,3] + Pin[13,6]*dVel_dTheta[0,0] + Pin[13,7]*dVel_dTheta[0,1] + Pin[13,8]*dVel_dTheta[0,2] + Pin[13,9]*dVel_dAccelBias[0,0]
    x103 = Pin[14,10]*dVel_dAccelBias[0,1] + Pin[14,11]*dVel_dAccelBias[0,2] + Pin[14,3] + Pin[14,6]*dVel_dTheta[0,0] + Pin[14,7]*dVel_dTheta[0,1] + Pin[14,8]*dVel_dTheta[0,2] + Pin[14,9]*dVel_dAccelBias[0,0]
    x104 = dVel_dAccelBias[1,0]*x73 + dVel_dAccelBias[1,1]*x74 + dVel_dAccelBias[1,2]*x75 + dVel_dTheta[1,0]*x76 + dVel_dTheta[1,1]*x77 + dVel_dTheta[1,2]*x78 + x27
    x105 = dVel_dAccelBias[1,0]*x80 + dVel_dAccelBias[1,1]*x81 + dVel_dAccelBias[1,2]*x82 + dVel_dTheta[1,0]*x83 + dVel_dTheta[1,1]*x84 + dVel_dTheta[1,2]*x85 + x30
    x106 = dVel_dAccelBias[1,0]*x87 + dVel_dAccelBias[1,1]*x88 + dVel_dAccelBias[1,2]*x89 + dVel_dTheta[1,0]*x90 + dVel_dTheta[1,1]*x91 + dVel_dTheta[1,2]*x92 + x33
    x107 = dVel_dAccelBias[1,0]*x94 + dVel_dAccelBias[1,1]*x95 + dVel_dAccelBias[1,2]*x96 + dVel_dTheta[1,0]*x97 + dVel_dTheta[1,1]*x98 + dVel_dTheta[1,2]*x99 + x36
    x108 = Pin[12,10]*dVel_dAccelBias[1,1] + Pin[12,11]*dVel_dAccelBias[1,2] + Pin[12,4] + Pin[12,6]*dVel_dTheta[1,0] + Pin[12,7]*dVel_dTheta[1,1] + Pin[12,8]*dVel_dTheta[1,2] + Pin[12,9]*dVel_dAccelBias[1,0]
    x109 = Pin[13,10]*dVel_dAccelBias[1,1] + Pin[13,11]*dVel_dAccelBias[1,2] + Pin[13,4] + Pin[13,6]*dVel_dTheta[1,0] + Pin[13,7]*dVel_dTheta[1,1] + Pin[13,8]*dVel_dTheta[1,2] + Pin[13,9]*dVel_dAccelBias[1,0]
    x110 = Pin[14,10]*dVel_dAccelBias[1,1] + Pin[14,11]*dVel_dAccelBias[1,2] + Pin[14,4] + Pin[14,6]*dVel_dTheta[1,0] + Pin[14,7]*dVel_dTheta[1,1] + Pin[14,8]*dVel_dTheta[1,2] + Pin[14,9]*dVel_dAccelBias[1,0]
    x111 = dVel_dAccelBias[2,0]*x80 + dVel_dAccelBias[2,1]*x81 + dVel_dAccelBias[2,2]*x82 + dVel_dTheta[2,0]*x83 + dVel_dTheta[2,1]*x84 + dVel_dTheta[2,2]*x85 + x49
    x112 = dVel_dAccelBias[2,0]*x87 + dVel_dAccelBias[2,1]*x88 + dVel_dAccelBias[2,2]*x89 + dVel_dTheta[2,0]*x90 + dVel_dTheta[2,1]*x91 + dVel_dTheta[2,2]*x92 + x52
    x113 = dVel_dAccelBias[2,0]*x94 + dVel_dAccelBias[2,1]*x95 + dVel_dAccelBias[2,2]*x96 + dVel_dTheta[2,0]*x97 + dVel_dTheta[2,1]*x98 + dVel_dTheta[2,2]*x99 + x55
    x114 = Pin[12,10]*dVel_dAccelBias[2,1] + Pin[12,11]*dVel_dAccelBias[2,2] + Pin[12,5] + Pin[12,6]*dVel_dTheta[2,0] + Pin[12,7]*dVel_dTheta[2,1] + Pin[12,8]*dVel_dTheta[2,2] + Pin[12,9]*dVel_dAccelBias[2,0]
    x115 = Pin[13,10]*dVel_dAccelBias[2,1] + Pin[13,11]*dVel_dAccelBias[2,2] + Pin[13,5] + Pin[13,6]*dVel_dTheta[2,0] + Pin[13,7]*dVel_dTheta[2,1] + Pin[13,8]*dVel_dTheta[2,2] + Pin[13,9]*dVel_dAccelBias[2,0]
    x116 = Pin[14,10]*dVel_dAccelBias[2,1] + Pin[14,11]*dVel_dAccelBias[2,2] + Pin[14,5] + Pin[14,6]*dVel_dTheta[2,0] + Pin[14,7]*dVel_dTheta[2,1] + Pin[14,8]*dVel_dTheta[2,2] + Pin[14,9]*dVel_dAccelBias[2,0]
    x117 = -Pin[12,12]*dt + Pin[12,6]*dTheta_dTheta[0,0] + Pin[12,7]*dTheta_dTheta[0,1] + Pin[12,8]*dTheta_dTheta[0,2]
    x118 = -Pin[13,12]*dt
    x119 = Pin[12,6]*dTheta_dTheta[1,0] + Pin[12,7]*dTheta_dTheta[1,1] + Pin[12,8]*dTheta_dTheta[1,2] + x118
    x120 = dTheta_dTheta[0,0]*x90 + dTheta_dTheta[0,1]*x91 + dTheta_dTheta[0,2]*x92 - dt*x119
    x121 = -Pin[14,12]*dt
    x122 = Pin[12,6]*dTheta_dTheta[2,0] + Pin[12,7]*dTheta_dTheta[2,1] + Pin[12,8]*dTheta_dTheta[2,2] + x121
    x123 = dTheta_dTheta[0,0]*x97 + dTheta_dTheta[0,1]*x98 + dTheta_dTheta[0,2]*x99 - dt*x122
    x124 = Pin[13,6]*dTheta_dTheta[0,0] + Pin[13,7]*dTheta_dTheta[0,1] + Pin[13,8]*dTheta_dTheta[0,2] + x118
    x125 = Pin[14,6]*dTheta_dTheta[0,0] + Pin[14,7]*dTheta_dTheta[0,1] + Pin[14,8]*dTheta_dTheta[0,2] + x121
    x126 = -Pin[13,13]*dt + Pin[13,6]*dTheta_dTheta[1,0] + Pin[13,7]*dTheta_dTheta[1,1] + Pin[13,8]*dTheta_dTheta[1,2]
    x127 = -Pin[14,13]*dt
    x128 = Pin[13,6]*dTheta_dTheta[2,0] + Pin[13,7]*dTheta_dTheta[2,1] + Pin[13,8]*dTheta_dTheta[2,2] + x127
    x129 = dTheta_dTheta[1,0]*x97 + dTheta_dTheta[1,1]*x98 + dTheta_dTheta[1,2]*x99 - dt*x128
    x130 = Pin[14,6]*dTheta_dTheta[1,0] + Pin[14,7]*dTheta_dTheta[1,1] + Pin[14,8]*dTheta_dTheta[1,2] + x127
    x131 = -Pin[14,14]*dt + Pin[14,6]*dTheta_dTheta[2,0] + Pin[14,7]*dTheta_dTheta[2,1] + Pin[14,8]*dTheta_dTheta[2,2]

    Pnew[0,0] = Pin[0,0] + Pin[3,0]*dt + dt*(Pin[3,0] + Pin[3,3]*dt)
    Pnew[0,1] = x0
    Pnew[0,2] = x1
    Pnew[0,3] = x3
    Pnew[0,4] = x5
    Pnew[0,5] = x7
    Pnew[0,6] = x10
    Pnew[0,7] = x13
    Pnew[0,8] = x16
    Pnew[0,9] = x17
    Pnew[0,10] = x18
    Pnew[0,11] = x19
    Pnew[0,12] = x20
    Pnew[0,13] = x21
    Pnew[0,14] = x22
    Pnew[1,0] = x0
    Pnew[1,1] = Pin[1,1] + Pin[4,1]*dt + dt*(Pin[4,1] + Pin[4,4]*dt)
    Pnew[1,2] = x23
    Pnew[1,3] = x24
    Pnew[1,4] = x26
    Pnew[1,5] = x28
    Pnew[1,6] = x31
    Pnew[1,7] = x34
    Pnew[1,8] = x37
    Pnew[1,9] = x38
    Pnew[1,10] = x39
    Pnew[1,11] = x40
    Pnew[1,12] = x41
    Pnew[1,13] = x42
    Pnew[1,14] = x43
    Pnew[2,0] = x1
    Pnew[2,1] = x23
    Pnew[2,2] = Pin[2,2] + Pin[5,2]*dt + dt*(Pin[5,2] + Pin[5,5]*dt)
    Pnew[2,3] = x44
    Pnew[2,4] = x45
    Pnew[2,5] = x47
    Pnew[2,6] = x50
    Pnew[2,7] = x53
    Pnew[2,8] = x56
    Pnew[2,9] = x57
    Pnew[2,10] = x58
    Pnew[2,11] = x59
    Pnew[2,12] = x60
    Pnew[2,13] = x61
    Pnew[2,14] = x62
    Pnew[3,0] = x3
    Pnew[3,1] = x24
    Pnew[3,2] = x44
    Pnew[3,3] = dVel_dAccelBias[0,0]*x63 + dVel_dAccelBias[0,1]*x64 + dVel_dAccelBias[0,2]*x65 + dVel_dTheta[0,0]*(Pin[10,6]*dVel_dAccelBias[0,1] + Pin[11,6]*dVel_dAccelBias[0,2] + Pin[6,3] + Pin[6,6]*dVel_dTheta[0,0] + Pin[7,6]*dVel_dTheta[0,1] + Pin[8,6]*dVel_dTheta[0,2] + Pin[9,6]*dVel_dAccelBias[0,0]) + dVel_dTheta[0,1]*(Pin[10,7]*dVel_dAccelBias[0,1] + Pin[11,7]*dVel_dAccelBias[0,2] + Pin[7,3] + Pin[7,6]*dVel_dTheta[0,0] + Pin[7,7]*dVel_dTheta[0,1] + Pin[8,7]*dVel_dTheta[0,2] + Pin[9,7]*dVel_dAccelBias[0,0]) + dVel_dTheta[0,2]*(Pin[10,8]*dVel_dAccelBias[0,1] + Pin[11,8]*dVel_dAccelBias[0,2] + Pin[8,3] + Pin[8,6]*dVel_dTheta[0,0] + Pin[8,7]*dVel_dTheta[0,1] + Pin[8,8]*dVel_dTheta[0,2] + Pin[9,8]*dVel_dAccelBias[0,0]) + x2
    Pnew[3,4] = x72
    Pnew[3,5] = x79
    Pnew[3,6] = x86
    Pnew[3,7] = x93
    Pnew[3,8] = x100
    Pnew[3,9] = x63
    Pnew[3,10] = x64
    Pnew[3,11] = x65
    Pnew[3,12] = x101
    Pnew[3,13] = x102
    Pnew[3,14] = x103
    Pnew[4,0] = x5
    Pnew[4,1] = x26
    Pnew[4,2] = x45
    Pnew[4,3] = x72
    Pnew[4,4] = dVel_dAccelBias[1,0]*x66 + dVel_dAccelBias[1,1]*x67 + dVel_dAccelBias[1,2]*x68 + dVel_dTheta[1,0]*x69 + dVel_dTheta[1,1]*x70 + dVel_dTheta[1,2]*x71 + x25
    Pnew[4,5] = x104
    Pnew[4,6] = x105
    Pnew[4,7] = x106
    Pnew[4,8] = x107
    Pnew[4,9] = x66
    Pnew[4,10] = x67
    Pnew[4,11] = x68
    Pnew[4,12] = x108
    Pnew[4,13] = x109
    Pnew[4,14] = x110
    Pnew[5,0] = x7
    Pnew[5,1] = x28
    Pnew[5,2] = x47
    Pnew[5,3] = x79
    Pnew[5,4] = x104
    Pnew[5,5] = dVel_dAccelBias[2,0]*x73 + dVel_dAccelBias[2,1]*x74 + dVel_dAccelBias[2,2]*x75 + dVel_dTheta[2,0]*x76 + dVel_dTheta[2,1]*x77 + dVel_dTheta[2,2]*x78 + x46
    Pnew[5,6] = x111
    Pnew[5,7] = x112
    Pnew[5,8] = x113
    Pnew[5,9] = x73
    Pnew[5,10] = x74
    Pnew[5,11] = x75
    Pnew[5,12] = x114
    Pnew[5,13] = x115
    Pnew[5,14] = x116
    Pnew[6,0] = x10
    Pnew[6,1] = x31
    Pnew[6,2] = x50
    Pnew[6,3] = x86
    Pnew[6,4] = x105
    Pnew[6,5] = x111
    Pnew[6,6] = dTheta_dTheta[0,0]*x83 + dTheta_dTheta[0,1]*x84 + dTheta_dTheta[0,2]*x85 - dt*x117
    Pnew[6,7] = x120
    Pnew[6,8] = x123
    Pnew[6,9] = x80
    Pnew[6,10] = x81
    Pnew[6,11] = x82
    Pnew[6,12] = x117
    Pnew[6,13] = x124
    Pnew[6,14] = x125
    Pnew[7,0] = x13
    Pnew[7,1] = x34
    Pnew[7,2] = x53
    Pnew[7,3] = x93
    Pnew[7,4] = x106
    Pnew[7,5] = x112
    Pnew[7,6] = x120
    Pnew[7,7] = dTheta_dTheta[1,0]*x90 + dTheta_dTheta[1,1]*x91 + dTheta_dTheta[1,2]*x92 - dt*x126
    Pnew[7,8] = x129
    Pnew[7,9] = x87
    Pnew[7,10] = x88
    Pnew[7,11] = x89
    Pnew[7,12] = x119
    Pnew[7,13] = x126
    Pnew[7,14] = x130
    Pnew[8,0] = x16
    Pnew[8,1] = x37
    Pnew[8,2] = x56
    Pnew[8,3] = x100
    Pnew[8,4] = x107
    Pnew[8,5] = x113
    Pnew[8,6] = x123
    Pnew[8,7] = x129
    Pnew[8,8] = dTheta_dTheta[2,0]*x97 + dTheta_dTheta[2,1]*x98 + dTheta_dTheta[2,2]*x99 - dt*x131
    Pnew[8,9] = x94
    Pnew[8,10] = x95
    Pnew[8,11] = x96
    Pnew[8,12] = x122
    Pnew[8,13] = x128
    Pnew[8,14] = x131
    Pnew[9,0] = x17
    Pnew[9,1] = x38
    Pnew[9,2] = x57
    Pnew[9,3] = x63
    Pnew[9,4] = x66
    Pnew[9,5] = x73
    Pnew[9,6] = x80
    Pnew[9,7] = x87
    Pnew[9,8] = x94
    Pnew[9,9] = Pin[9,9]
    Pnew[9,10] = Pin[10,9]
    Pnew[9,11] = Pin[11,9]
    Pnew[9,12] = Pin[12,9]
    Pnew[9,13] = Pin[13,9]
    Pnew[9,14] = Pin[14,9]
    Pnew[10,0] = x18
    Pnew[10,1] = x39
    Pnew[10,2] = x58
    Pnew[10,3] = x64
    Pnew[10,4] = x67
    Pnew[10,5] = x74
    Pnew[10,6] = x81
    Pnew[10,7] = x88
    Pnew[10,8] = x95
    Pnew[10,9] = Pin[10,9]
    Pnew[10,10] = Pin[10,10]
    Pnew[10,11] = Pin[11,10]
    Pnew[10,12] = Pin[12,10]
    Pnew[10,13] = Pin[13,10]
    Pnew[10,14] = Pin[14,10]
    Pnew[11,0] = x19
    Pnew[11,1] = x40
    Pnew[11,2] = x59
    Pnew[11,3] = x65
    Pnew[11,4] = x68
    Pnew[11,5] = x75
    Pnew[11,6] = x82
    Pnew[11,7] = x89
    Pnew[11,8] = x96
    Pnew[11,9] = Pin[11,9]
    Pnew[11,10] = Pin[11,10]
    Pnew[11,11] = Pin[11,11]
    Pnew[11,12] = Pin[12,11]
    Pnew[11,13] = Pin[13,11]
    Pnew[11,14] = Pin[14,11]
    Pnew[12,0] = x20
    Pnew[12,1] = x41
    Pnew[12,2] = x60
    Pnew[12,3] = x101
    Pnew[12,4] = x108
    Pnew[12,5] = x114
    Pnew[12,6] = x117
    Pnew[12,7] = x119
    Pnew[12,8] = x122
    Pnew[12,9] = Pin[12,9]
    Pnew[12,10] = Pin[12,10]
    Pnew[12,11] = Pin[12,11]
    Pnew[12,12] = Pin[12,12]
    Pnew[12,13] = Pin[13,12]
    Pnew[12,14] = Pin[14,12]
    Pnew[13,0] = x21
    Pnew[13,1] = x42
    Pnew[13,2] = x61
    Pnew[13,3] = x102
    Pnew[13,4] = x109
    Pnew[13,5] = x115
    Pnew[13,6] = x124
    Pnew[13,7] = x126
    Pnew[13,8] = x128
    Pnew[13,9] = Pin[13,9]
    Pnew[13,10] = Pin[13,10]
    Pnew[13,11] = Pin[13,11]
    Pnew[13,12] = Pin[13,12]
    Pnew[13,13] = Pin[13,13]
    Pnew[13,14] = Pin[14,13]
    Pnew[14,0] = x22
    Pnew[14,1] = x43
    Pnew[14,2] = x62
    Pnew[14,3] = x103
    Pnew[14,4] = x110
    Pnew[14,5] = x116
    Pnew[14,6] = x125
    Pnew[14,7] = x130
    Pnew[14,8] = x131
    Pnew[14,9] = Pin[14,9]
    Pnew[14,10] = Pin[14,10]
    Pnew[14,11] = Pin[14,11]
    Pnew[14,12] = Pin[14,12]
    Pnew[14,13] = Pin[14,13]
    Pnew[14,14] = Pin[14,14]
