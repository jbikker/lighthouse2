/* buildcontext.h - Copyright 2019 Utrecht University
   
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   
   	   http://www.apache.org/licenses/LICENSE-2.0
   
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#pragma once

#include "Recast.h"		  // rcContext, RC_MAX_TIMERS, rcLogCategory, rcTimerLabel
#include "rendersystem.h" // Timer, memset

namespace lighthouse2 {

//  +-----------------------------------------------------------------------------+
//  |  BuildContext                                                               |
//  |  An implementation of the virtual native Recast logging class.        LH2'19|
//  +-----------------------------------------------------------------------------+
class BuildContext : public rcContext
{
	float m_startTime[RC_MAX_TIMERS];
	float m_accTime[RC_MAX_TIMERS];

	static const int MAX_MESSAGES = 1000;
	const char* m_messages[MAX_MESSAGES];
	int m_messageCount;
	static const int TEXT_POOL_SIZE = 8000;
	char m_textPool[TEXT_POOL_SIZE];
	int m_textPoolSize;
	Timer timer;

public:
	BuildContext() : m_messageCount(0), m_textPoolSize(0)
	{
		memset(m_messages, 0, sizeof(char*) * MAX_MESSAGES);
		resetTimers();
	};

	// Dumps the log to stdout.
	std::string GetBuildStats();
	// Returns number of log messages.
	int getLogCount() const { return m_messageCount; };
	// Returns log message text.
	const char* getLogText(const int i) const { return m_messages[i] + 1; };

protected:
	// Virtual functions for custom implementations.
	virtual void doResetLog() { m_messageCount = 0; m_textPoolSize = 0; };
	virtual void doLog(const rcLogCategory category, const char* msg, const int len);
	virtual void doResetTimers() { for (int i = 0; i < RC_MAX_TIMERS; ++i) m_accTime[i] = -1; };
	virtual void doStartTimer(const rcTimerLabel label) { m_startTime[label] = timer.elapsed(); };
	virtual void doStopTimer(const rcTimerLabel label);
	virtual int doGetAccumulatedTime(const rcTimerLabel label) const { return m_accTime[label] * 1000000.0f; };
};

//  +-----------------------------------------------------------------------------+
//  |  BuildContext::doLog                                                        |
//  |  Virtual function called by rcContext::log                                 |
//  |  Adds messages to the queue.                                          LH2'19|
//  +-----------------------------------------------------------------------------+
void BuildContext::doLog(const rcLogCategory category, const char* msg, const int len)
{
	if (!len) return;
	if (m_messageCount >= MAX_MESSAGES)
		return;
	char* dst = &m_textPool[m_textPoolSize];
	int n = TEXT_POOL_SIZE - m_textPoolSize;
	if (n < 2)
		return;
	char* cat = dst;
	char* text = dst + 1;
	const int maxtext = n - 1;
	// Store category
	*cat = (char)category;
	// Store message
	const int count = rcMin(len + 1, maxtext);
	memcpy(text, msg, count);
	text[count - 1] = '\0';
	m_textPoolSize += 1 + count;
	m_messages[m_messageCount++] = dst;
}

//  +-----------------------------------------------------------------------------+
//  |  BuildContext::doStopTimer                                                  |
//  |  Virtual function called by rcContext::stopTimer                            |
//  |  Stops the timer and calculates the passed time.                      LH2'19|
//  +-----------------------------------------------------------------------------+
void BuildContext::doStopTimer(const rcTimerLabel label)
{
	const float endTime = timer.elapsed();
	const float deltaTime = endTime - m_startTime[label];
	if (m_accTime[label] == -1)
		m_accTime[label] = deltaTime;
	else
		m_accTime[label] += deltaTime;
}

//  +-----------------------------------------------------------------------------+
//  |  BuildContext::dumpLog                                                      |
//  |  Prints all logged messages to stdout.                                LH2'19|
//  +-----------------------------------------------------------------------------+
std::string BuildContext::GetBuildStats()
{
	// Logging timestamp for Detour data (not included in Recast logging)
	int t = getAccumulatedTime(RC_TIMER_TEMP);
	log(RC_LOG_PROGRESS, "- Creating Detour data:\t%.2fms\t(%.1f%%)",
		t / 1000.0f, t*100.0f / getAccumulatedTime(RC_TIMER_TOTAL));

	// Print messages
	std::string buildStats;
	const int TAB_STOPS[4] = { 28, 36, 44, 52 };
	for (int i = 0; i <= m_messageCount; ++i)
	{
		if (i == m_messageCount - 2) i++; // skip [last - 2]
		else if (i == m_messageCount) i -= 2; // print skipped log last
		const char* msg = m_messages[i] + 1;
		int n = 0;
		while (*msg)
		{
			if (*msg == '\t')
			{
				int count = 1;
				for (int j = 0; j < 4; ++j)
					if (n < TAB_STOPS[j])
					{
						count = TAB_STOPS[j] - n;
						break;
					}
				while (--count) { buildStats += ' '; n++; }
			}
			else if (*msg == '%')
			{
				buildStats += "%%";
			}
			else
			{
				buildStats += *msg;
				n++;
			}
			msg++;
		}
		buildStats += '\n';
		if (i == m_messageCount - 2) break; // stop loop
	}
	resetLog();
	return buildStats;
}

} // namespace lighthouse2

// EOF