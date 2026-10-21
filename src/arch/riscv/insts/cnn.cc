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

#include "arch/riscv/insts/cnn.hh"

#include <sstream>
#include <string>

#include "arch/riscv/utility.hh"

namespace gem5
{

namespace RiscvISA
{

std::string
CnnOp::generateDisassembly(Addr pc, const loader::SymbolTable *symtab) const
{
    // cnn.dot8 and cnn.dwmac also read rd, which comes first in the source
    // list; rs1 and rs2 are always the last two sources.
    std::stringstream ss;
    ss << mnemonic << ' ' << registerName(destRegIdx(0)) << ", " <<
        registerName(srcRegIdx(_numSrcRegs - 2)) << ", " <<
        registerName(srcRegIdx(_numSrcRegs - 1));
    // cnn.dwmac carries the byte-pair selector p in funct3.
    if (machInst.opcode5 == 0x02 && machInst.funct7 == 0x01)
        ss << ", " << (uint32_t)machInst.funct3;
    return ss.str();
}

} // namespace RiscvISA
} // namespace gem5
