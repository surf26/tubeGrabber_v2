#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""列出当前接的所有 Orbbec 相机及序列号，方便多相机时把序列号填进 config.py 的 CAMERA_SERIAL。"""
from pyorbbecsdk import Context


def main():
    dev_list = Context().query_devices()
    count = dev_list.get_count()
    if count == 0:
        print("没有检测到 Orbbec 相机，检查 USB 连接。")
        return
    print(f"检测到 {count} 台 Orbbec 相机：")
    for i in range(count):
        info = dev_list.get_device_by_index(i).get_device_info()
        print(f"  [{i}] 名称={info.get_name()}  序列号={info.get_serial_number()}")
    if count > 1:
        print("\n多台相机：把你要用于标定的那台的序列号填进 config.py 的 CAMERA_SERIAL。")


if __name__ == "__main__":
    main()
