// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The consumer fixture's classes: Parent stands for a class of a provider kext,
// which the fixture never defines, and Consumer overrides one of its methods
// and appends one.
#pragma once
class Parent {
public:
    virtual ~Parent();
    virtual int inherited(int);
    virtual int changed(int);
    virtual int unused(int);

protected:
    unsigned long long storage[3];
};
class Consumer : public Parent {
public:
    ~Consumer() override;
    int changed(int) override;
    virtual int appended(int);
};
