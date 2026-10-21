/*
 * Copyright (c) 2026 Yuri Nunes
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Custom int8 CNN instructions of the riscv-cnn-bench project
 * (https://github.com/YuriMZ-Nunes/riscv-cnn-bench, docs/custom_instructions.md).
 *
 * All of them are R-type, register-only and never trap:
 *   custom-0 (0x0B): cnn.dot8 (funct7 0, funct3 0), cnn.max8 (funct7 0,
 *                    funct3 1), cnn.dwmac p (funct7 1, funct3 = p in 0..3)
 *   custom-1 (0x2B): cnn.rqnt (funct7 0, funct3 0), cnn.rqnt.relu
 *                    (funct7 1, funct3 0)
 * Other funct3/funct7 combinations are reserved and decode as unknown.
 *
 * The functions below mirror the C model in benchmarks/common/cnn_insn.h of
 * the project, which is the reference semantics.
 */

#ifndef __ARCH_RISCV_INSTS_CNN_HH__
#define __ARCH_RISCV_INSTS_CNN_HH__

#include <cstdint>
#include <string>

#include "arch/riscv/insts/static_inst.hh"

namespace gem5
{

namespace RiscvISA
{

class CnnOp : public RiscvStaticInst
{
  protected:
    using RiscvStaticInst::RiscvStaticInst;

    std::string generateDisassembly(
        Addr pc, const loader::SymbolTable *symtab) const override;
};

namespace cnn
{

// Signed byte i of a packed register (byte 0 in bits 7:0).
inline int32_t
lane8(uint64_t reg, unsigned i)
{
    return (int8_t)(uint8_t)(reg >> (8 * i));
}

// cnn.dot8: sext32(rd[31:0] + sum of 8 int8 x int8 products), mod 2^32.
inline uint64_t
dot8(uint64_t rd, uint64_t rs1, uint64_t rs2)
{
    uint32_t sum = (uint32_t)rd;
    for (unsigned i = 0; i < 8; i++)
        sum += (uint32_t)(lane8(rs1, i) * lane8(rs2, i));
    return (uint64_t)(int64_t)(int32_t)sum;
}

// cnn.dwmac p: each int32 lane j of rd accumulates byte 2p+j of rs1 * rs2.
inline uint64_t
dwmac(uint64_t rd, uint64_t rs1, uint64_t rs2, unsigned p)
{
    uint64_t out = 0;
    for (unsigned j = 0; j < 2; j++) {
        unsigned k = 2 * (p & 3) + j;
        uint32_t lane = (uint32_t)(rd >> (32 * j));
        lane += (uint32_t)(lane8(rs1, k) * lane8(rs2, k));
        out |= (uint64_t)lane << (32 * j);
    }
    return out;
}

// cnn.max8: signed maximum, byte by byte.
inline uint64_t
max8(uint64_t rs1, uint64_t rs2)
{
    uint64_t out = 0;
    for (unsigned i = 0; i < 8; i++) {
        int32_t x = lane8(rs1, i);
        int32_t y = lane8(rs2, i);
        out |= (uint64_t)(uint8_t)(x > y ? x : y) << (8 * i);
    }
    return out;
}

// cnn.rqnt[.relu]: sat8(relu?((a * M0 + 2^(30+s)) >> (31+s))), with
// a = rs1[31:0], M0 = rs2[31:0] and s = rs2[36:32]; result sign-extended.
inline uint64_t
rqnt(uint64_t rs1, uint64_t rs2, bool relu)
{
    int64_t a = (int32_t)(uint32_t)rs1;
    int64_t m0 = (int32_t)(uint32_t)rs2;
    unsigned s = (rs2 >> 32) & 31;
    // |a * M0| <= 2^62 and the rounding term is <= 2^61: no overflow.
    int64_t y = (a * m0 + ((int64_t)1 << (30 + s))) >> (31 + s);
    int64_t lo = relu ? 0 : -128;
    if (y < lo)
        y = lo;
    if (y > 127)
        y = 127;
    return (uint64_t)y;
}

} // namespace cnn

} // namespace RiscvISA
} // namespace gem5

#endif // __ARCH_RISCV_INSTS_CNN_HH__
